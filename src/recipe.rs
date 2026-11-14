//! Architecture recipe for Microsoft Phi-3 and Phi-4 (`phi3`, `phi4`).
//!
//! Executes authentic multi-quant GGUF forward passes across `F32`, `F16`, `BF16`, `Q4_0`,
//! `Q5_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` tensors with fused `QKV`/`Gate-Up` row slicing,
//! partial/full `NeoX` `LongRoPE` (`short_factor`, `long_factor`, `attn_factor`), layer-major
//! `0..=pos` [`KvCacheView`] causal GQA attention, and tied/untied LM head projection.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, TensorView,
};

use crate::quant::{
    decode_1d_norm, dequantize_row_to_f32, dot_product_f32, gqa_attention_f32, project_matvec,
    resolve_2d_dims, rmsnorm_f32, swiglu_layer_f32,
};

/// `RoPE` scaling strategies supported by Phi-3 and Phi-4 architectures.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RopeScalingConfig {
    /// Standard unscaled rotary position embeddings.
    #[default]
    None,
    /// Uniform linear context length extension factor.
    Linear {
        /// Linear frequency divisor (`> 0.0`).
        factor: f32,
    },
    /// Dual-band `LongRoPE` scaling switching beyond `original_max_position_embeddings`.
    LongRope {
        /// Per-frequency divisors active within the native pretraining context.
        short_factor: Vec<f32>,
        /// Per-frequency divisors active beyond `original_max_position_embeddings`.
        long_factor: Vec<f32>,
        /// Native context boundary separating short and long factor regimes.
        original_max_position_embeddings: usize,
    },
}

/// Architecture recipe for Microsoft `Phi-3` (`phi3`) and `Phi-4` (`phi4`) models.
#[derive(Debug, Clone, PartialEq)]
pub struct Phi3Recipe {
    /// Canonical architecture identifier (`"phi3"` or `"phi4"`).
    pub architecture: &'static str,
    /// Number of transformer decoder layers.
    pub num_layers: usize,
    /// Model embedding dimension (`d_model`).
    pub hidden_dim: usize,
    /// Intermediate `SwiGLU` MLP projection width (`d_ff`).
    pub ffn_dim: usize,
    /// Number of query attention heads.
    pub n_heads_q: usize,
    /// Number of key/value attention heads (for MHA or GQA).
    pub n_heads_kv: usize,
    /// Scalar dimension per attention head.
    pub head_dim: usize,
    /// Optional partial `RoPE` dimension count (`<= head_dim`).
    pub rotary_dim: Option<usize>,
    /// Model vocabulary size.
    pub vocab_size: usize,
    /// `RMSNorm` numerical stability epsilon.
    pub rms_eps: f32,
    /// Base `RoPE` inverse frequency theta.
    pub rope_theta: f32,
    /// Multiplicative temperature factor applied to `LongRoPE` `(cos, sin)` pairs.
    pub rope_attn_factor: f32,
    /// Native context window before `LongRoPE` switches to `long_factor`.
    pub original_max_position_embeddings: usize,
    /// Optional sliding-window attention span in tokens.
    pub sliding_window: Option<usize>,
    /// Active rotary position scaling configuration.
    pub rope_scaling: RopeScalingConfig,
}

impl Default for Phi3Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Phi3Recipe {
    /// Creates a `Phi-3.5-mini` (`phi3`, `3.8B`, MHA `32:32`, `32064` vocab) recipe preset.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "phi3",
            num_layers: 32,
            hidden_dim: 3072,
            ffn_dim: 8192,
            n_heads_q: 32,
            n_heads_kv: 32,
            head_dim: 96,
            rotary_dim: None,
            vocab_size: 32_064,
            rms_eps: 1.0e-5,
            rope_theta: 10_000.0,
            rope_attn_factor: 1.0,
            original_max_position_embeddings: 4096,
            sliding_window: None,
            rope_scaling: RopeScalingConfig::None,
        }
    }

    /// Creates a `Phi-4` (`phi4`, `14B`, GQA `40:10`, `100352` vocab) recipe preset.
    #[must_use]
    pub const fn new_phi4() -> Self {
        Self {
            architecture: "phi4",
            num_layers: 40,
            hidden_dim: 5120,
            ffn_dim: 17_920,
            n_heads_q: 40,
            n_heads_kv: 10,
            head_dim: 128,
            rotary_dim: None,
            vocab_size: 100_352,
            rms_eps: 1.0e-5,
            rope_theta: 250_000.0,
            rope_attn_factor: 1.0,
            original_max_position_embeddings: 16_384,
            sliding_window: None,
            rope_scaling: RopeScalingConfig::None,
        }
    }

    /// Creates a `Phi-4-mini` (`phi3`, `3.8B`, GQA `32:8`, `200064` vocab) recipe preset.
    #[must_use]
    pub const fn new_phi4_mini() -> Self {
        Self {
            architecture: "phi3",
            num_layers: 32,
            hidden_dim: 3072,
            ffn_dim: 8192,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 96,
            rotary_dim: None,
            vocab_size: 200_064,
            rms_eps: 1.0e-5,
            rope_theta: 10_000.0,
            rope_attn_factor: 1.0,
            original_max_position_embeddings: 4096,
            sliding_window: None,
            rope_scaling: RopeScalingConfig::None,
        }
    }

    /// Attaches a [`RopeScalingConfig`] to this recipe.
    #[must_use]
    pub fn with_rope_scaling(mut self, scaling: RopeScalingConfig) -> Self {
        self.rope_scaling = scaling;
        self
    }

    /// Configures an explicit partial rotary dimension (`<= head_dim`).
    #[must_use]
    pub const fn with_rotary_dim(mut self, rotary_dim: Option<usize>) -> Self {
        self.rotary_dim = rotary_dim;
        self
    }

    /// Sets the multiplicative `LongRoPE` attention factor (`cos`/`sin` magnitude scaling).
    #[must_use]
    pub const fn with_attn_factor(mut self, attn_factor: f32) -> Self {
        self.rope_attn_factor = attn_factor;
        self
    }

    /// Sets an optional sliding-window attention token span.
    #[must_use]
    pub const fn with_sliding_window(mut self, sliding_window: Option<usize>) -> Self {
        self.sliding_window = sliding_window;
        self
    }

    /// Returns the effective rotary dimension per head (`rotary_dim` when set, else `head_dim`).
    #[must_use]
    pub const fn effective_rotary_dim(&self) -> usize {
        match self.rotary_dim {
            Some(dim) if dim > 0 => dim,
            _ => self.head_dim,
        }
    }

    /// Returns `true` when `position` exceeds the native training window and selects `long_factor`.
    #[must_use]
    pub const fn is_long_context(&self, position: usize) -> bool {
        let orig_ctx = match &self.rope_scaling {
            RopeScalingConfig::LongRope {
                original_max_position_embeddings,
                ..
            } if *original_max_position_embeddings > 0 => *original_max_position_embeddings,
            _ => self.original_max_position_embeddings,
        };
        orig_ctx > 0 && position > orig_ctx
    }

    /// Returns the active frequency factor slice when [`RopeScalingConfig::LongRope`] is active.
    #[must_use]
    pub fn active_rope_factors(&self, position: usize) -> Option<&[f32]> {
        let RopeScalingConfig::LongRope {
            short_factor,
            long_factor,
            ..
        } = &self.rope_scaling
        else {
            return None;
        };
        if self.is_long_context(position) {
            Some(long_factor.as_slice())
        } else {
            Some(short_factor.as_slice())
        }
    }

    /// Hydrates a [`Phi3Recipe`] dynamically from parsed GGUF [`ModelMetadata`].
    ///
    /// Supports `phi3` (`Phi-3.5-mini`, `Phi-4-mini`) and `phi4` (`Phi-4-14B`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `metadata.architecture()` is not `"phi3"` or
    /// `"phi4"`, if core dimensions are zero or non-divisible (`n_heads_q % n_heads_kv != 0`), or
    /// if `rotary_dim > head_dim` or is odd.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, K3Error> {
        let arch = metadata.architecture();
        let canonical_arch = match arch {
            "phi3" => "phi3",
            "phi4" => "phi4",
            other => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: other.to_owned(),
                    reason: format!("expected phi3 or phi4 architecture, got '{other}'"),
                });
            }
        };

        let base = if canonical_arch == "phi4" {
            Self::new_phi4()
        } else {
            Self::new()
        };

        let num_layers = metadata.layer_count();
        let hidden_dim = metadata.embedding_dim();
        let n_heads_q = metadata.head_count();
        let n_heads_kv = metadata.head_count_kv();
        if num_layers == 0 || hidden_dim == 0 || n_heads_q == 0 || n_heads_kv == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: canonical_arch.to_owned(),
                reason: format!(
                    "non-zero Phi metadata dimensions required (layers={num_layers}, \
                     hidden={hidden_dim}, n_heads_q={n_heads_q}, n_heads_kv={n_heads_kv})"
                ),
            });
        }
        if n_heads_q % n_heads_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: canonical_arch.to_owned(),
                reason: format!(
                    "GQA head_count ({n_heads_q}) must be divisible by head_count_kv ({n_heads_kv})"
                ),
            });
        }

        let head_dim = metadata
            .key_length()
            .filter(|&k| k > 0)
            .or_else(|| metadata.value_length().filter(|&v| v > 0))
            .unwrap_or(hidden_dim / n_heads_q);
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: canonical_arch.to_owned(),
                reason: format!("Phi head_dim ({head_dim}) must be non-zero and even"),
            });
        }

        let rotary_dim = metadata.rope_dimension_count().filter(|&d| d > 0);
        if let Some(rot) = rotary_dim
            && (rot > head_dim || rot % 2 != 0)
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: canonical_arch.to_owned(),
                reason: format!(
                    "rope.dimension_count ({rot}) must be even and <= head_dim ({head_dim})"
                ),
            });
        }

        let ffn_dim = metadata
            .feed_forward_length()
            .filter(|&f| f > 0)
            .unwrap_or(base.ffn_dim);

        let inferred_vocab = metadata.find_tensor("token_embd.weight").and_then(|t| {
            let s = t.shape();
            if s.len() == 2 && s[0] == hidden_dim && s[1] > 0 {
                Some(s[1])
            } else if s.len() == 2 && s[1] == hidden_dim && s[0] > 0 {
                Some(s[0])
            } else {
                None
            }
        });
        let vocab_size = metadata
            .vocab_size()
            .filter(|&v| v > 0)
            .or(inferred_vocab)
            .unwrap_or(base.vocab_size);

        let original_max_position_embeddings = if metadata.context_length() > 0 {
            metadata.context_length()
        } else {
            base.original_max_position_embeddings
        };

        Ok(Self {
            architecture: canonical_arch,
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            rotary_dim,
            vocab_size,
            rms_eps: metadata.rms_norm_eps().unwrap_or(base.rms_eps),
            rope_theta: metadata.rope_freq_base().unwrap_or(base.rope_theta),
            rope_attn_factor: 1.0,
            original_max_position_embeddings,
            sliding_window: metadata.sliding_window().filter(|&w| w > 0),
            rope_scaling: RopeScalingConfig::None,
        })
    }

    /// Hydrates a complete [`Phi3Recipe`] from a parsed [`GgufFile`], extracting `LongRoPE`
    /// `short_factor`, `long_factor`, `attn_factor`, and `original_context_length` overrides.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if metadata invariants or `LongRoPE` array lengths
    /// fail validation (`short_factor.len() == rotary_dim / 2`, finite positive factors).
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(gguf.metadata())?;
        let arch = recipe.architecture;

        if let Some(orig_val) = lookup_phi_kv(gguf, arch, "rope.scaling.original_context_length")
            && let Some(orig_ctx) = orig_val.as_usize()
            && orig_ctx > 0
        {
            recipe.original_max_position_embeddings = orig_ctx;
        }

        if let Some(attn_val) = lookup_phi_kv(gguf, arch, "rope.scaling.attn_factor") {
            let Some(attn) = attn_val.as_f32() else {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: "rope.scaling.attn_factor must be a finite positive float".to_owned(),
                });
            };
            if !attn.is_finite() || attn <= 0.0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!("rope.scaling.attn_factor ({attn}) must be finite and > 0.0"),
                });
            }
            recipe.rope_attn_factor = attn;
        }

        let scaling_type =
            lookup_phi_kv(gguf, arch, "rope.scaling.type").and_then(GgufValue::as_str);
        let short_opt = lookup_phi_kv(gguf, arch, "rope.scaling.short_factor")
            .map(|v| extract_positive_f32_vec(arch, "rope.scaling.short_factor", v))
            .transpose()?;
        let long_opt = lookup_phi_kv(gguf, arch, "rope.scaling.long_factor")
            .map(|v| extract_positive_f32_vec(arch, "rope.scaling.long_factor", v))
            .transpose()?;

        let is_longrope =
            scaling_type == Some("longrope") || short_opt.is_some() || long_opt.is_some();

        if is_longrope {
            let short_factor = short_opt.ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "LongRoPE scaling requires rope.scaling.short_factor".to_owned(),
            })?;
            let long_factor = long_opt.ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "LongRoPE scaling requires rope.scaling.long_factor".to_owned(),
            })?;

            let expected_len = recipe.effective_rotary_dim() / 2;
            if short_factor.len() != expected_len || long_factor.len() != expected_len {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!(
                        "LongRoPE factor length mismatch: expected {expected_len} (rotary_dim/2), \
                         got short={} and long={}",
                        short_factor.len(),
                        long_factor.len()
                    ),
                });
            }

            recipe.rope_scaling = RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings: recipe.original_max_position_embeddings,
            };
        } else if scaling_type == Some("linear")
            && let Some(factor_val) = lookup_phi_kv(gguf, arch, "rope.scaling.factor")
        {
            let factor = factor_val
                .as_f32()
                .filter(|&f| f.is_finite() && f > 0.0)
                .ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: "linear rope.scaling.factor must be finite and > 0.0".to_owned(),
                })?;
            recipe.rope_scaling = RopeScalingConfig::Linear { factor };
        }

        Ok(recipe)
    }

    fn validate_runtime_geometry(&self) -> Result<(), K3Error> {
        if self.num_layers == 0
            || self.hidden_dim == 0
            || self.ffn_dim == 0
            || self.n_heads_q == 0
            || self.n_heads_kv == 0
            || self.head_dim == 0
            || self.vocab_size == 0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "Phi3Recipe runtime dimensions must all be non-zero".to_owned(),
            });
        }
        if self.n_heads_q % self.n_heads_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "n_heads_q ({}) must be divisible by n_heads_kv ({})",
                    self.n_heads_q, self.n_heads_kv
                ),
            });
        }
        if self.head_dim % 2 != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!("head_dim ({}) must be even", self.head_dim),
            });
        }
        let rot = self.effective_rotary_dim();
        if rot == 0 || rot > self.head_dim || rot % 2 != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "rotary_dim ({rot}) must be positive, even, and <= head_dim ({})",
                    self.head_dim
                ),
            });
        }
        if !self.rope_attn_factor.is_finite() || self.rope_attn_factor <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "rope_attn_factor ({}) must be finite and > 0.0",
                    self.rope_attn_factor
                ),
            });
        }
        match &self.rope_scaling {
            RopeScalingConfig::None => Ok(()),
            RopeScalingConfig::Linear { factor } => {
                if !factor.is_finite() || *factor <= 0.0 {
                    Err(K3Error::CapabilityMismatch {
                        architecture: self.architecture.to_owned(),
                        reason: format!("linear rope factor ({factor}) must be finite and > 0.0"),
                    })
                } else {
                    Ok(())
                }
            }
            RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                ..
            } => {
                let half = rot / 2;
                if short_factor.len() != half || long_factor.len() != half {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: self.architecture.to_owned(),
                        reason: format!(
                            "LongRoPE factors must have length {half} (rotary_dim/2), \
                             got short={} and long={}",
                            short_factor.len(),
                            long_factor.len()
                        ),
                    });
                }
                if short_factor
                    .iter()
                    .chain(long_factor.iter())
                    .any(|&f| !f.is_finite() || f <= 0.0)
                {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: self.architecture.to_owned(),
                        reason: "LongRoPE factors must all be finite and > 0.0".to_owned(),
                    });
                }
                Ok(())
            }
        }
    }

    fn apply_rope_and_attn_factor(&self, heads_buf: &mut [f32], num_heads: usize, abs_pos: usize) {
        let rotary_dim = self.effective_rotary_dim();
        let half = rotary_dim / 2;
        if half == 0 {
            return;
        }
        let factors = self.active_rope_factors(abs_pos);
        let linear_factor = match self.rope_scaling {
            RopeScalingConfig::Linear { factor } => factor,
            _ => 1.0,
        };
        let attn_scale = self.rope_attn_factor;
        #[expect(clippy::cast_precision_loss, reason = "rotary dimension fits in f32")]
        let rot_f32 = rotary_dim as f32;
        #[expect(clippy::cast_precision_loss, reason = "sequence position fits in f32")]
        let pos_f32 = abs_pos as f32;

        for h in 0..num_heads {
            let head = &mut heads_buf[h * self.head_dim..(h + 1) * self.head_dim];
            for i in 0..half {
                let ext = factors.map_or(linear_factor, |f| f[i]);
                #[expect(clippy::cast_precision_loss, reason = "pair index fits in f32")]
                let exponent = (2.0_f32 * (i as f32)) / rot_f32;
                let inv_freq = 1.0_f32 / (self.rope_theta.powf(exponent) * ext);
                let (sin_a, cos_a) = (pos_f32 * inv_freq).sin_cos();
                let c = cos_a * attn_scale;
                let s = sin_a * attn_scale;
                let x0 = head[i];
                let x1 = head[i + half];
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        }
    }

    fn forward_token(
        &self,
        token_id: u32,
        abs_pos: usize,
        weights: &dyn ModelWeights,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut ForwardWorkspace,
    ) -> Result<(), K3Error> {
        let synthetic_mode = weights.tensor_count() == 0;
        let hidden = self.hidden_dim;

        if synthetic_mode {
            synthesize_embedding(token_id, &mut workspace.hidden);
        } else {
            let embd_view = weights.tensor("token_embd.weight")?;
            dequantize_row_to_f32(&embd_view, token_id as usize, hidden, &mut workspace.hidden)?;
        }

        for layer in 0..self.num_layers {
            if synthetic_mode {
                self.forward_synthetic_layer(layer, abs_pos, kv_view, workspace)?;
            } else {
                self.forward_authentic_layer(layer, abs_pos, weights, kv_view, workspace)?;
            }
        }

        Ok(())
    }

    fn forward_synthetic_layer(
        &self,
        layer: usize,
        abs_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut ForwardWorkspace,
    ) -> Result<(), K3Error> {
        let total_kv = self.n_heads_kv * self.head_dim;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        rmsnorm_f32(
            &mut workspace.normed,
            &workspace.hidden,
            &workspace.unit_norm,
            self.rms_eps,
        )?;
        synthesize_qkv(
            layer,
            &workspace.normed,
            &mut workspace.q,
            &mut workspace.k,
            &mut workspace.v,
        );
        self.apply_rope_and_attn_factor(&mut workspace.q, self.n_heads_q, abs_pos);
        self.apply_rope_and_attn_factor(&mut workspace.k, self.n_heads_kv, abs_pos);

        {
            let (k_slot, v_slot) = kv_view.write_pos_range(layer, abs_pos)?;
            k_slot.copy_from_slice(&workspace.k);
            v_slot.copy_from_slice(&workspace.v);
        }

        let (k_valid, v_valid) = kv_view.valid_kv_range(layer, abs_pos + 1)?;
        let valid_seq_len = k_valid.len() / total_kv;
        gqa_attention_f32(
            &mut workspace.attn_out,
            &workspace.q,
            k_valid,
            v_valid,
            self.n_heads_q,
            self.n_heads_kv,
            self.head_dim,
            valid_seq_len,
            attn_scale,
        )?;

        accumulate_folded_proj(&workspace.attn_out, &mut workspace.hidden, 0.25);
        rmsnorm_f32(
            &mut workspace.normed,
            &workspace.hidden,
            &workspace.unit_norm,
            self.rms_eps,
        )?;
        synthesize_gate_up(
            self.ffn_dim,
            &workspace.normed,
            &mut workspace.gate,
            &mut workspace.up,
        );
        swiglu_layer_f32(&mut workspace.ffn_act, &workspace.gate, &workspace.up)?;
        accumulate_folded_proj(&workspace.ffn_act, &mut workspace.hidden, 0.25);
        Ok(())
    }

    fn forward_authentic_layer(
        &self,
        layer: usize,
        abs_pos: usize,
        weights: &dyn ModelWeights,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut ForwardWorkspace,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        let total_q = self.n_heads_q * self.head_dim;
        let total_kv = self.n_heads_kv * self.head_dim;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        let attn_norm_name = format!("blk.{layer}.attn_norm.weight");
        let norm_view = weights.tensor(&attn_norm_name)?;
        let norm_w = decode_1d_norm(&norm_view, hidden)?;
        rmsnorm_f32(
            &mut workspace.normed,
            &workspace.hidden,
            &norm_w,
            self.rms_eps,
        )?;

        project_layer_qkv(
            layer,
            hidden,
            total_q,
            total_kv,
            weights,
            &workspace.normed,
            &mut workspace.q,
            &mut workspace.k,
            &mut workspace.v,
        )?;

        self.apply_rope_and_attn_factor(&mut workspace.q, self.n_heads_q, abs_pos);
        self.apply_rope_and_attn_factor(&mut workspace.k, self.n_heads_kv, abs_pos);

        {
            let (k_slot, v_slot) = kv_view.write_pos_range(layer, abs_pos)?;
            k_slot.copy_from_slice(&workspace.k);
            v_slot.copy_from_slice(&workspace.v);
        }

        let (k_valid, v_valid) = kv_view.valid_kv_range(layer, abs_pos + 1)?;
        let valid_seq_len = k_valid.len() / total_kv;
        gqa_attention_f32(
            &mut workspace.attn_out,
            &workspace.q,
            k_valid,
            v_valid,
            self.n_heads_q,
            self.n_heads_kv,
            self.head_dim,
            valid_seq_len,
            attn_scale,
        )?;

        let o_name = format!("blk.{layer}.attn_output.weight");
        let o_view = weights.tensor(&o_name)?;
        project_matvec(&mut workspace.proj_hidden, &o_view, &workspace.attn_out)?;
        for (h, &delta) in workspace
            .hidden
            .iter_mut()
            .zip(workspace.proj_hidden.iter())
        {
            *h += delta;
        }

        let ffn_norm_w = load_ffn_norm(layer, hidden, weights)?;
        rmsnorm_f32(
            &mut workspace.normed,
            &workspace.hidden,
            &ffn_norm_w,
            self.rms_eps,
        )?;

        project_layer_mlp_up(
            layer,
            hidden,
            self.ffn_dim,
            weights,
            &workspace.normed,
            &mut workspace.gate,
            &mut workspace.up,
        )?;
        swiglu_layer_f32(&mut workspace.ffn_act, &workspace.gate, &workspace.up)?;

        let down_name = format!("blk.{layer}.ffn_down.weight");
        let down_view = weights.tensor(&down_name)?;
        project_matvec(&mut workspace.proj_hidden, &down_view, &workspace.ffn_act)?;
        for (h, &delta) in workspace
            .hidden
            .iter_mut()
            .zip(workspace.proj_hidden.iter())
        {
            *h += delta;
        }
        Ok(())
    }

    fn write_lm_logits(
        &self,
        weights: &dyn ModelWeights,
        hidden_state: &[f32],
        normed_out: &mut [f32],
        logits_out: &mut [f32],
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        if logits_out.len() < self.vocab_size {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "output.weight".to_owned(),
                offset: 0,
                byte_len: self.vocab_size * size_of::<f32>(),
                buffer_len: size_of_val(logits_out),
                reason: format!(
                    "logits buffer len {} smaller than vocab_size {}",
                    logits_out.len(),
                    self.vocab_size
                ),
            });
        }

        if weights.tensor_count() == 0 {
            let unit = vec![1.0_f32; hidden];
            rmsnorm_f32(normed_out, hidden_state, &unit, self.rms_eps)?;
            let norm_sum: f32 = normed_out.iter().sum();
            logits_out.fill(0.0);
            for (v, slot) in logits_out.iter_mut().enumerate().take(self.vocab_size) {
                let mut probe = vec![0.0_f32; hidden];
                synthesize_embedding(u32::try_from(v).unwrap_or(0), &mut probe);
                *slot = dot_product_f32(normed_out, &probe) + norm_sum * 1.0e-3;
            }
            return Ok(());
        }

        let out_norm_view = weights.tensor("output_norm.weight")?;
        let out_norm_w = decode_1d_norm(&out_norm_view, hidden)?;
        rmsnorm_f32(normed_out, hidden_state, &out_norm_w, self.rms_eps)?;

        let lm_head_view = match weights.tensor("output.weight") {
            Ok(view) => view,
            Err(K3Error::TensorBoundsError { .. }) => weights.tensor("token_embd.weight")?,
            Err(other) => return Err(other),
        };
        let (lm_vocab, _) = resolve_2d_dims(&lm_head_view, hidden)?;
        if lm_vocab != self.vocab_size {
            return Err(K3Error::TensorBoundsError {
                tensor_name: lm_head_view.name().to_owned(),
                offset: lm_head_view.offset(),
                byte_len: lm_head_view.byte_len(),
                buffer_len: lm_head_view.as_bytes().len(),
                reason: format!(
                    "LM head row count {lm_vocab} does not match recipe vocab_size {}",
                    self.vocab_size
                ),
            });
        }

        if logits_out.len() > lm_vocab {
            logits_out[lm_vocab..].fill(0.0);
        }
        project_matvec(&mut logits_out[..lm_vocab], &lm_head_view, normed_out)
    }

    fn resolve_kv_geometry(&self, ctx: &ExecutionContext<'_>) -> Result<KvGeometry, K3Error> {
        if let Some(geom) = ctx.kv_geometry() {
            if geom.num_layers != self.num_layers
                || geom.n_heads_kv != self.n_heads_kv
                || geom.head_dim != self.head_dim
            {
                return Err(K3Error::CapabilityMismatch {
                    architecture: self.architecture.to_owned(),
                    reason: format!(
                        "ExecutionContext KvGeometry ({}, {}, {}) != recipe ({}, {}, {})",
                        geom.num_layers,
                        geom.n_heads_kv,
                        geom.head_dim,
                        self.num_layers,
                        self.n_heads_kv,
                        self.head_dim
                    ),
                });
            }
            return Ok(geom);
        }
        KvGeometry::new(
            self.num_layers,
            self.n_heads_kv,
            self.head_dim,
            ctx.n_ctx(),
            self.sliding_window,
        )
    }
}

struct ForwardWorkspace {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    proj_hidden: Vec<f32>,
    unit_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_act: Vec<f32>,
}

impl ForwardWorkspace {
    fn new(recipe: &Phi3Recipe) -> Self {
        let total_q = recipe.n_heads_q * recipe.head_dim;
        let total_kv = recipe.n_heads_kv * recipe.head_dim;
        Self {
            hidden: vec![0.0_f32; recipe.hidden_dim],
            normed: vec![0.0_f32; recipe.hidden_dim],
            proj_hidden: vec![0.0_f32; recipe.hidden_dim],
            unit_norm: vec![1.0_f32; recipe.hidden_dim],
            q: vec![0.0_f32; total_q],
            k: vec![0.0_f32; total_kv],
            v: vec![0.0_f32; total_kv],
            attn_out: vec![0.0_f32; total_q],
            gate: vec![0.0_f32; recipe.ffn_dim],
            up: vec![0.0_f32; recipe.ffn_dim],
            ffn_act: vec![0.0_f32; recipe.ffn_dim],
        }
    }
}

fn lookup_phi_kv<'a>(
    gguf: &'a GgufFile<'a>,
    arch: &str,
    suffix: &str,
) -> Option<&'a GgufValue<'a>> {
    let primary = format!("{arch}.{suffix}");
    if let Some(val) = gguf.metadata_value(&primary) {
        return Some(val);
    }
    if arch != "phi3" {
        let phi3_key = format!("phi3.{suffix}");
        if let Some(val) = gguf.metadata_value(&phi3_key) {
            return Some(val);
        }
    }
    if arch != "phi4" {
        let phi4_key = format!("phi4.{suffix}");
        if let Some(val) = gguf.metadata_value(&phi4_key) {
            return Some(val);
        }
    }
    None
}

fn extract_positive_f32_vec(
    arch: &str,
    key: &str,
    val: &GgufValue<'_>,
) -> Result<Vec<f32>, K3Error> {
    let Some(arr) = val.as_array() else {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{key} must be a numeric GGUF array"),
        });
    };
    let factors: Vec<f32> = match arr {
        GgufArray::Float32(items) => items.clone(),
        #[expect(
            clippy::cast_possible_truncation,
            reason = "LongRoPE scaling factors are small positive constants representable in f32"
        )]
        GgufArray::Float64(items) => items.iter().map(|&v| v as f32).collect(),
        _ => {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{key} must be a Float32 or Float64 array"),
            });
        }
    };
    if factors.is_empty() || factors.iter().any(|&f| !f.is_finite() || f <= 0.0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{key} must be non-empty with finite positive factors"),
        });
    }
    Ok(factors)
}

fn project_row_range(
    out: &mut [f32],
    tensor: &TensorView<'_>,
    expected_cols: usize,
    row_start: usize,
    row_count: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    if out.len() != row_count || input.len() != expected_cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "row-slice projection buffer mismatch (out={}, row_count={}, in={}, cols={expected_cols})",
                out.len(),
                row_count,
                input.len()
            ),
        });
    }
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("expected 2D tensor, got shape {shape:?}"),
        });
    }
    if shape[0] == expected_cols {
        let sliced = tensor.slice_rows(row_start, row_count)?;
        return project_matvec(out, &sliced, input);
    }
    if shape[1] == expected_cols {
        let total_rows = shape[0];
        let end_row = row_start.saturating_add(row_count);
        if end_row > total_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: tensor.byte_len(),
                buffer_len: tensor.as_bytes().len(),
                reason: format!("row range {row_start}..{end_row} exceeds row count {total_rows}"),
            });
        }
        let mut row_buf = vec![0.0_f32; expected_cols];
        for (i, slot) in out.iter_mut().enumerate() {
            dequantize_row_to_f32(tensor, row_start + i, expected_cols, &mut row_buf)?;
            *slot = dot_product_f32(&row_buf, input);
        }
        return Ok(());
    }
    Err(K3Error::TensorBoundsError {
        tensor_name: tensor.name().to_owned(),
        offset: tensor.offset(),
        byte_len: tensor.byte_len(),
        buffer_len: tensor.as_bytes().len(),
        reason: format!("2D shape {shape:?} does not match expected cols {expected_cols}"),
    })
}

#[expect(clippy::too_many_arguments, reason = "explicit QKV slice projections")]
fn project_layer_qkv(
    layer: usize,
    hidden: usize,
    total_q: usize,
    total_kv: usize,
    weights: &dyn ModelWeights,
    normed: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<(), K3Error> {
    let qkv_name = format!("blk.{layer}.attn_qkv.weight");
    match weights.tensor(&qkv_name) {
        Ok(qkv_view) => {
            let (rows, _) = resolve_2d_dims(&qkv_view, hidden)?;
            let expected_rows = total_q + 2 * total_kv;
            if rows != expected_rows {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: qkv_name,
                    offset: qkv_view.offset(),
                    byte_len: qkv_view.byte_len(),
                    buffer_len: qkv_view.as_bytes().len(),
                    reason: format!(
                        "fused attn_qkv rows {rows} != expected total_q + 2*total_kv ({expected_rows})"
                    ),
                });
            }
            project_row_range(q_out, &qkv_view, hidden, 0, total_q, normed)?;
            project_row_range(k_out, &qkv_view, hidden, total_q, total_kv, normed)?;
            project_row_range(
                v_out,
                &qkv_view,
                hidden,
                total_q + total_kv,
                total_kv,
                normed,
            )
        }
        Err(K3Error::TensorBoundsError { .. }) => {
            let q_view = weights.tensor(&format!("blk.{layer}.attn_q.weight"))?;
            let k_view = weights.tensor(&format!("blk.{layer}.attn_k.weight"))?;
            let v_view = weights.tensor(&format!("blk.{layer}.attn_v.weight"))?;
            project_matvec(q_out, &q_view, normed)?;
            project_matvec(k_out, &k_view, normed)?;
            project_matvec(v_out, &v_view, normed)
        }
        Err(other) => Err(other),
    }
}

fn project_layer_mlp_up(
    layer: usize,
    hidden: usize,
    ffn_dim: usize,
    weights: &dyn ModelWeights,
    normed: &[f32],
    gate_out: &mut [f32],
    up_out: &mut [f32],
) -> Result<(), K3Error> {
    let fused_name = format!("blk.{layer}.ffn_up.weight");
    let up_view = weights.tensor(&fused_name)?;
    let (rows, _) = resolve_2d_dims(&up_view, hidden)?;
    if rows == 2 * ffn_dim {
        project_row_range(gate_out, &up_view, hidden, 0, ffn_dim, normed)?;
        project_row_range(up_out, &up_view, hidden, ffn_dim, ffn_dim, normed)
    } else if rows == ffn_dim {
        let gate_view = weights.tensor(&format!("blk.{layer}.ffn_gate.weight"))?;
        project_matvec(gate_out, &gate_view, normed)?;
        project_matvec(up_out, &up_view, normed)
    } else {
        Err(K3Error::TensorBoundsError {
            tensor_name: fused_name,
            offset: up_view.offset(),
            byte_len: up_view.byte_len(),
            buffer_len: up_view.as_bytes().len(),
            reason: format!(
                "ffn_up rows {rows} matches neither fused 2*ffn_dim ({}) nor unfused ffn_dim ({ffn_dim})",
                2 * ffn_dim
            ),
        })
    }
}

fn load_ffn_norm(
    layer: usize,
    hidden: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let primary = format!("blk.{layer}.ffn_norm.weight");
    match weights.tensor(&primary) {
        Ok(view) => decode_1d_norm(&view, hidden),
        Err(K3Error::TensorBoundsError { .. }) => {
            let alt = format!("blk.{layer}.post_attention_norm.weight");
            let view = weights.tensor(&alt)?;
            decode_1d_norm(&view, hidden)
        }
        Err(other) => Err(other),
    }
}

fn synthesize_embedding(token_id: u32, out: &mut [f32]) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "synthetic lane index fits in f32"
    )]
    let hidden_f = out.len().max(1) as f32;
    let tok_f = f32::from(u16::try_from(token_id % 997).unwrap_or(0)) * 0.03125;
    for (idx, slot) in out.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "dimension index fits in f32")]
        let phase = (idx as f32) / hidden_f + tok_f;
        *slot = (phase * std::f32::consts::PI).sin() * 0.5;
    }
}

fn synthesize_qkv(layer: usize, normed: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    #[expect(clippy::cast_precision_loss, reason = "layer index fits in f32")]
    let layer_bias = ((layer % 7) as f32) * 0.05 + 0.75;
    for (i, slot) in q.iter_mut().enumerate() {
        *slot = normed[i % normed.len()] * layer_bias;
    }
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = normed[(i + 1) % normed.len()] * (layer_bias * 0.9);
    }
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = normed[(i + 2) % normed.len()] * (layer_bias * 0.85);
    }
}

fn synthesize_gate_up(ffn_dim: usize, normed: &[f32], gate: &mut [f32], up: &mut [f32]) {
    let n = normed.len().max(1);
    for i in 0..ffn_dim {
        gate[i] = normed[i % n] * 0.5;
        up[i] = normed[(i + 3) % n] * 0.5;
    }
}

fn accumulate_folded_proj(src: &[f32], dst: &mut [f32], scale: f32) {
    let d_len = dst.len().max(1);
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector fold ratio fits cleanly in f32"
    )]
    let fold = (src.len() / d_len).max(1) as f32;
    let inv = scale / fold;
    for (i, &val) in src.iter().enumerate() {
        dst[i % d_len] += val * inv;
    }
}

impl ArchRecipe for Phi3Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn recipe_version(&self) -> &'static str {
        "1.0.0"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q5_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let h_q = if metadata.head_count() > 0 {
            metadata.head_count()
        } else {
            self.n_heads_q
        };
        let h_kv = if metadata.head_count_kv() > 0 {
            metadata.head_count_kv()
        } else {
            self.n_heads_kv
        };
        if h_q == 0 || h_kv == 0 || h_q % h_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: format!("GQA head_count ({h_q}) not divisible by head_count_kv ({h_kv})"),
            });
        }
        self.validate_runtime_geometry()?;

        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0
                && let Ok(view) = weights.tensor(desc.name())
                && !self.supports_quant(view.quant())
            {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    view.name(),
                    view.quant(),
                ));
            }
        }

        if weights.tensor_count() > 0 {
            for required in ["token_embd.weight", "output_norm.weight"] {
                let view = weights.tensor(required)?;
                if !self.supports_quant(view.quant()) {
                    return Err(K3Error::unsupported_tensor_quant(
                        self.architecture,
                        view.name(),
                        view.quant(),
                    ));
                }
            }
            if let Ok(out_view) = weights.tensor("output.weight")
                && !self.supports_quant(out_view.quant())
            {
                return Err(K3Error::unsupported_tensor_quant(
                    self.architecture,
                    out_view.name(),
                    out_view.quant(),
                ));
            }
            for layer in 0..self.num_layers {
                for suffix in [
                    "attn_norm.weight",
                    "attn_output.weight",
                    "ffn_up.weight",
                    "ffn_down.weight",
                ] {
                    let name = format!("blk.{layer}.{suffix}");
                    let view = weights.tensor(&name)?;
                    if !self.supports_quant(view.quant()) {
                        return Err(K3Error::unsupported_tensor_quant(
                            self.architecture,
                            view.name(),
                            view.quant(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_runtime_geometry()?;
        if ctx.prompt_tokens().is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "prefill_step requires at least one prompt token".to_owned(),
            });
        }

        let prompt_tokens: Vec<u32> = ctx.prompt_tokens().to_vec();
        let start_pos = ctx.position();
        let end_pos =
            start_pos
                .checked_add(prompt_tokens.len())
                .ok_or(K3Error::ContextLengthExceeded {
                    requested_tokens: usize::MAX,
                    max_context_tokens: ctx.n_ctx(),
                })?;
        if end_pos > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let geom = self.resolve_kv_geometry(ctx)?;
        let mut workspace = ForwardWorkspace::new(self);
        {
            let (logits_buf, kv_buf) = ctx.buffers_mut();
            let mut kv_view = KvCacheView::new(kv_buf, geom)?;

            for (offset, &token_id) in prompt_tokens.iter().enumerate() {
                self.forward_token(
                    token_id,
                    start_pos + offset,
                    weights,
                    &mut kv_view,
                    &mut workspace,
                )?;
            }

            self.write_lm_logits(
                weights,
                &workspace.hidden,
                &mut workspace.normed,
                logits_buf,
            )?;
        }
        ctx.advance_position(prompt_tokens.len())?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_runtime_geometry()?;
        let token_id = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "decode_step requires a decode_token in ExecutionContext".to_owned(),
            })?;
        let abs_pos = ctx.position();
        if abs_pos >= ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: abs_pos.saturating_add(1),
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let geom = self.resolve_kv_geometry(ctx)?;
        let mut workspace = ForwardWorkspace::new(self);
        {
            let (logits_buf, kv_buf) = ctx.buffers_mut();
            let mut kv_view = KvCacheView::new(kv_buf, geom)?;

            self.forward_token(token_id, abs_pos, weights, &mut kv_view, &mut workspace)?;
            self.write_lm_logits(
                weights,
                &workspace.hidden,
                &mut workspace.normed,
                logits_buf,
            )?;
        }
        ctx.advance_position(1)?;
        Ok(())
    }
}
