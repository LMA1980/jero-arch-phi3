//! Architecture recipe for Microsoft Phi-3 and Phi-4 (`phi3`, `phi4`).
//!
//! Executes authentic multi-quant GGUF forward passes across `F32`, `F16`, `BF16`, `Q4_0`,
//! `Q5_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` tensors with fused `QKV`/`Gate-Up` row slicing,
//! partial/full `NeoX` `LongRoPE` (`short_factor`, `long_factor`, `attn_factor`), layer-major
//! `0..=pos` [`KvCacheView`] causal GQA attention, and tied/untied LM head projection.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, ScratchScope, ScratchWorkspace, StopCriteria,
    TensorView,
};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{QuantizedActQ8, QuantizedActQ8View, RopeType, apply_rope_with_cos_sin};

#[cfg(feature = "with-kernels")]
use crate::quant::{SharedQ8Input, project_matvec_shared_q8};
use crate::quant::{
    decode_1d_norm_into, dequantize_row_to_f32, dot_product_f32, gqa_attention_with_scratch_f32,
    gqa_prefill_causal_f32, project_lm_head_parallel, project_matvec, resolve_2d_dims,
    rmsnorm_batch_f32, rmsnorm_f32, swiglu_layer_f32, with_2d_rows_view,
};
#[cfg(feature = "with-kernels")]
use crate::repack::get_or_repack_layer;
use crate::repack::{Phi3LayerDescriptor, Phi3LayerRepack, Phi3ModelRepack};

/// Canonical padded vocabulary size for `Phi-3` / `Phi-3.5-mini` models (`32,064`).
pub const PHI35_PADDED_VOCAB_SIZE: usize = 32_064;
/// Canonical vocabulary size for `Phi-4-mini` (`o200k`, `200,064`).
pub const PHI4_MINI_VOCAB_SIZE: usize = 200_064;
/// Canonical vocabulary size for `Phi-4` (`14B`, `100,352`).
pub const PHI4_14B_VOCAB_SIZE: usize = 100_352;

/// Chat-template and end-of-turn delimiter dialect used by Phi-3 / Phi-4 GGUF models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiChatDialect {
    /// Phi-3 / Phi-3.5 / Phi-4-mini `<|user|>...<|end|>\n<|assistant|>` turn framing (`<|end|>` stop).
    Phi3End,
    /// Phi-4 `<|im_start|>role<|im_sep|>content<|im_end|>` turn framing (`<|im_sep|>` / `<|im_end|>` stop).
    Phi4ImSep,
}

/// Validated vocabulary bounds, chat dialect, and [`StopCriteria`]-compatible stop token IDs
/// extracted from Phi-3 / Phi-4 GGUF metadata and tensor descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiStopMetadata {
    /// Detected chat framing dialect.
    pub dialect: PhiChatDialect,
    /// Validated model vocabulary row count shared across embeddings, LM head, and logits buffers.
    pub vocab_size: usize,
    /// Entry count in `tokenizer.ggml.tokens`, when present.
    pub token_list_len: Option<usize>,
    /// Beginning-of-sequence token ID (`tokenizer.ggml.bos_token_id`), if present.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID (`tokenizer.ggml.eos_token_id`), if present.
    pub eos_token_id: Option<u32>,
    /// End-of-turn token ID (`tokenizer.ggml.eot_token_id`), if present.
    pub eot_token_id: Option<u32>,
    /// Token ID resolved for `"<|end|>"` in `tokenizer.ggml.tokens`, if present.
    pub end_token_id: Option<u32>,
    /// Token ID resolved for `"<|im_sep|>"` in `tokenizer.ggml.tokens`, if present.
    pub im_sep_token_id: Option<u32>,
    /// Token ID resolved for `"<|im_end|>"` in `tokenizer.ggml.tokens`, if present.
    pub im_end_token_id: Option<u32>,
    /// Token ID resolved for `"<|endoftext|>"` in `tokenizer.ggml.tokens`, if present.
    pub endoftext_token_id: Option<u32>,
    /// Deterministic, deduplicated stop token IDs passed directly to [`StopCriteria`].
    pub stop_token_ids: Vec<u32>,
    /// `true` when `output.weight` is absent and the LM head ties to `token_embd.weight`.
    pub has_tied_lm_head: bool,
}

impl PhiStopMetadata {
    /// Builds a [`StopCriteria`] referencing [`Self::stop_token_ids`] for sampling termination.
    #[must_use]
    pub fn stop_criteria(&self, max_tokens: usize, n_ctx: usize) -> StopCriteria<'_> {
        StopCriteria::new(&self.stop_token_ids, max_tokens, n_ctx)
    }

    /// Fails closed with [`K3Error::CapabilityMismatch`] if `logits.len()` diverges from
    /// [`Self::vocab_size`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] when `logits.len() != self.vocab_size`.
    pub fn validate_logits_slice(&self, architecture: &str, logits: &[f32]) -> Result<(), K3Error> {
        if logits.len() != self.vocab_size {
            return Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!(
                    "logits slice len {} diverges from validated GGUF vocab_size {}",
                    logits.len(),
                    self.vocab_size
                ),
            });
        }
        Ok(())
    }

    /// Returns `true` if `token_id` matches any resolved stop/EOT token ID.
    #[must_use]
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }
}

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

        let vocab_size =
            resolve_vocab_size_from_metadata(metadata, hidden_dim, base.vocab_size, arch)?;

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

        recipe.validate_gguf_contracts(gguf)?;
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

    fn fill_long_rope_cos_sin(&self, abs_pos: usize, cos_out: &mut [f32], sin_out: &mut [f32]) {
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

        for i in 0..half {
            let ext = factors.map_or(linear_factor, |f| f[i]);
            #[expect(clippy::cast_precision_loss, reason = "pair index fits in f32")]
            let exponent = (2.0_f32 * (i as f32)) / rot_f32;
            let inv_freq = 1.0_f32 / (self.rope_theta.powf(exponent) * ext);
            let (sin_a, cos_a) = (pos_f32 * inv_freq).sin_cos();
            cos_out[i] = cos_a * attn_scale;
            sin_out[i] = sin_a * attn_scale;
        }
    }

    #[cfg_attr(
        not(feature = "with-kernels"),
        expect(
            clippy::unnecessary_wraps,
            reason = "with-kernels path delegates to fallible apply_rope_with_cos_sin"
        )
    )]
    fn apply_rope_with_table(
        &self,
        heads_buf: &mut [f32],
        num_heads: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), K3Error> {
        let rotary_dim = self.effective_rotary_dim();
        let half = rotary_dim / 2;
        if half == 0 {
            return Ok(());
        }
        #[cfg(feature = "with-kernels")]
        if rotary_dim % 2 == 0 {
            return apply_rope_with_cos_sin(
                heads_buf,
                num_heads,
                self.head_dim,
                rotary_dim,
                RopeType::NeoX,
                cos,
                sin,
            );
        }
        for h in 0..num_heads {
            let head = &mut heads_buf[h * self.head_dim..(h + 1) * self.head_dim];
            for i in 0..half {
                let c = cos[i];
                let s = sin[i];
                let x0 = head[i];
                let x1 = head[i + half];
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        }
        Ok(())
    }

    fn apply_rope_pair_with_scratch(
        &self,
        q_buf: &mut [f32],
        k_buf: &mut [f32],
        abs_pos: usize,
        cos_scratch: &mut [f32],
        sin_scratch: &mut [f32],
    ) {
        let rotary_dim = self.effective_rotary_dim();
        let half = rotary_dim / 2;
        if half == 0 {
            return;
        }
        let cos = &mut cos_scratch[..half];
        let sin = &mut sin_scratch[..half];
        self.fill_long_rope_cos_sin(abs_pos, cos, sin);
        let _ = self.apply_rope_with_table(q_buf, self.n_heads_q, cos, sin);
        let _ = self.apply_rope_with_table(k_buf, self.n_heads_kv, cos, sin);
    }

    fn apply_rope_batch(
        &self,
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        m: usize,
        start_pos: usize,
        cos_scratch: &mut [f32],
        sin_scratch: &mut [f32],
    ) -> Result<(), K3Error> {
        let half = self.effective_rotary_dim() / 2;
        if half == 0 {
            return Ok(());
        }
        let total_q = self.n_heads_q * self.head_dim;
        let total_kv = self.n_heads_kv * self.head_dim;
        for m_idx in 0..m {
            let abs_pos = start_pos + m_idx;
            self.fill_long_rope_cos_sin(abs_pos, cos_scratch, sin_scratch);
            self.apply_rope_with_table(
                &mut q_batch[m_idx * total_q..(m_idx + 1) * total_q],
                self.n_heads_q,
                cos_scratch,
                sin_scratch,
            )?;
            self.apply_rope_with_table(
                &mut k_batch[m_idx * total_kv..(m_idx + 1) * total_kv],
                self.n_heads_kv,
                cos_scratch,
                sin_scratch,
            )?;
        }
        Ok(())
    }

    fn forward_token(
        &self,
        token_id: u32,
        abs_pos: usize,
        weights: &dyn ModelWeights,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut DecodeScratch<'_, '_>,
        repack_cache: Option<&Phi3ModelRepack>,
    ) -> Result<(), K3Error> {
        let synthetic_mode = weights.tensor_count() == 0;
        let hidden = self.hidden_dim;

        if synthetic_mode {
            if repack_cache.is_some() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: self.architecture.to_owned(),
                    reason: "repack cache cannot be used with empty synthetic weights".to_owned(),
                });
            }
            synthesize_embedding(token_id, workspace.hidden);
        } else {
            let embd_view = weights.tensor("token_embd.weight")?;
            dequantize_row_to_f32(&embd_view, token_id as usize, hidden, workspace.hidden)?;
        }

        for layer in 0..self.num_layers {
            if synthetic_mode {
                self.forward_synthetic_layer(layer, abs_pos, kv_view, workspace)?;
            } else if let Some(lr) = repack_cache.and_then(|cache| cache.layer(layer)) {
                self.forward_authentic_layer(
                    layer,
                    abs_pos,
                    weights,
                    Some(lr),
                    kv_view,
                    workspace,
                )?;
            } else {
                #[cfg(feature = "with-kernels")]
                let cached_layer = get_or_repack_layer(self, layer, weights)?;
                #[cfg(feature = "with-kernels")]
                let layer_repack = Some(cached_layer.as_ref());
                #[cfg(not(feature = "with-kernels"))]
                let layer_repack = None;
                self.forward_authentic_layer(
                    layer,
                    abs_pos,
                    weights,
                    layer_repack,
                    kv_view,
                    workspace,
                )?;
            }
        }

        Ok(())
    }

    fn forward_synthetic_layer(
        &self,
        layer: usize,
        abs_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut DecodeScratch<'_, '_>,
    ) -> Result<(), K3Error> {
        let total_kv = self.n_heads_kv * self.head_dim;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        workspace.norm_w.fill(1.0_f32);
        rmsnorm_f32(
            workspace.normed,
            workspace.hidden,
            workspace.norm_w,
            self.rms_eps,
        )?;
        synthesize_qkv(
            layer,
            workspace.normed,
            workspace.q,
            workspace.k,
            workspace.v,
        );
        self.apply_rope_pair_with_scratch(
            workspace.q,
            workspace.k,
            abs_pos,
            workspace.cos,
            workspace.sin,
        );

        {
            let (k_slot, v_slot) = kv_view.write_pos_range(layer, abs_pos)?;
            k_slot.copy_from_slice(workspace.k);
            v_slot.copy_from_slice(workspace.v);
        }

        let (k_valid, v_valid) = kv_view.valid_kv_range(layer, abs_pos + 1)?;
        let valid_seq_len = k_valid.len() / total_kv;
        gqa_attention_with_scratch_f32(
            workspace.attn_out,
            workspace.q,
            k_valid,
            v_valid,
            self.n_heads_q,
            self.n_heads_kv,
            self.head_dim,
            valid_seq_len,
            attn_scale,
            workspace.scores,
        )?;

        accumulate_folded_proj(workspace.attn_out, workspace.hidden, 0.25);
        rmsnorm_f32(
            workspace.normed,
            workspace.hidden,
            workspace.norm_w,
            self.rms_eps,
        )?;
        synthesize_gate_up(self.ffn_dim, workspace.normed, workspace.gate, workspace.up);
        swiglu_layer_f32(workspace.ffn_act, workspace.gate, workspace.up)?;
        accumulate_folded_proj(workspace.ffn_act, workspace.hidden, 0.25);
        Ok(())
    }

    fn forward_authentic_layer(
        &self,
        layer: usize,
        abs_pos: usize,
        weights: &dyn ModelWeights,
        layer_repack: Option<&Phi3LayerRepack>,
        kv_view: &mut KvCacheView<'_>,
        workspace: &mut DecodeScratch<'_, '_>,
    ) -> Result<(), K3Error> {
        let total_kv = self.n_heads_kv * self.head_dim;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (self.head_dim as f32).sqrt();
        let desc = Phi3LayerDescriptor::for_layer(layer);

        self.project_layer_attn_qkv_into_ws(&desc, weights, layer_repack, workspace)?;
        self.apply_rope_pair_with_scratch(
            workspace.q,
            workspace.k,
            abs_pos,
            workspace.cos,
            workspace.sin,
        );

        {
            let (k_slot, v_slot) = kv_view.write_pos_range(layer, abs_pos)?;
            k_slot.copy_from_slice(workspace.k);
            v_slot.copy_from_slice(workspace.v);
        }

        let (k_valid, v_valid) = kv_view.valid_kv_range(layer, abs_pos + 1)?;
        let valid_seq_len = k_valid.len() / total_kv;
        gqa_attention_with_scratch_f32(
            workspace.attn_out,
            workspace.q,
            k_valid,
            v_valid,
            self.n_heads_q,
            self.n_heads_kv,
            self.head_dim,
            valid_seq_len,
            attn_scale,
            workspace.scores,
        )?;

        Self::project_layer_attn_output_and_add(&desc, weights, layer_repack, workspace)?;
        self.project_layer_mlp_and_add(&desc, weights, layer_repack, workspace)
    }

    fn project_layer_attn_qkv_into_ws(
        &self,
        desc: &Phi3LayerDescriptor,
        weights: &dyn ModelWeights,
        layer_repack: Option<&Phi3LayerRepack>,
        workspace: &mut DecodeScratch<'_, '_>,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        let total_q = self.n_heads_q * self.head_dim;
        let total_kv = self.n_heads_kv * self.head_dim;
        if let Some(lr) = layer_repack {
            rmsnorm_f32(
                workspace.normed,
                workspace.hidden,
                lr.attn_norm(),
                self.rms_eps,
            )?;
            project_repacked_qkv(lr, workspace)
        } else {
            let norm_view = weights.tensor(desc.attn_norm())?;
            decode_1d_norm_into(&norm_view, hidden, workspace.norm_w)?;
            rmsnorm_f32(
                workspace.normed,
                workspace.hidden,
                workspace.norm_w,
                self.rms_eps,
            )?;
            project_layer_qkv(
                desc,
                hidden,
                total_q,
                total_kv,
                weights,
                workspace.normed,
                workspace.q,
                workspace.k,
                workspace.v,
            )
        }
    }

    fn project_layer_attn_output_and_add(
        desc: &Phi3LayerDescriptor,
        weights: &dyn ModelWeights,
        layer_repack: Option<&Phi3LayerRepack>,
        workspace: &mut DecodeScratch<'_, '_>,
    ) -> Result<(), K3Error> {
        if let Some(lr) = layer_repack {
            #[cfg(feature = "with-kernels")]
            {
                let q_act =
                    QuantizedActQ8View::quantize_row_in_scope(workspace.attn_out, workspace.scope)?;
                lr.wo().gemv_view_parallel(workspace.proj_hidden, &q_act)?;
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                lr.wo()
                    .gemv_parallel(workspace.proj_hidden, workspace.attn_out)?;
            }
        } else {
            let o_view = weights.tensor(desc.attn_output())?;
            project_matvec(workspace.proj_hidden, &o_view, workspace.attn_out)?;
        }
        add_residual_in_place(workspace.hidden, workspace.proj_hidden);
        Ok(())
    }

    fn project_layer_mlp_and_add(
        &self,
        desc: &Phi3LayerDescriptor,
        weights: &dyn ModelWeights,
        layer_repack: Option<&Phi3LayerRepack>,
        workspace: &mut DecodeScratch<'_, '_>,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        if let Some(lr) = layer_repack {
            rmsnorm_f32(
                workspace.normed,
                workspace.hidden,
                lr.ffn_norm(),
                self.rms_eps,
            )?;
            project_repacked_mlp_up(lr, workspace)?;
        } else {
            load_ffn_norm_into(desc, hidden, weights, workspace.norm_w)?;
            rmsnorm_f32(
                workspace.normed,
                workspace.hidden,
                workspace.norm_w,
                self.rms_eps,
            )?;
            project_layer_mlp_up(
                desc,
                hidden,
                self.ffn_dim,
                weights,
                workspace.normed,
                workspace.gate,
                workspace.up,
            )?;
        }
        swiglu_layer_f32(workspace.ffn_act, workspace.gate, workspace.up)?;
        if let Some(lr) = layer_repack {
            #[cfg(feature = "with-kernels")]
            {
                let q_act =
                    QuantizedActQ8View::quantize_row_in_scope(workspace.ffn_act, workspace.scope)?;
                lr.wdown()
                    .gemv_view_parallel(workspace.proj_hidden, &q_act)?;
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                lr.wdown()
                    .gemv_parallel(workspace.proj_hidden, workspace.ffn_act)?;
            }
        } else {
            let down_view = weights.tensor(desc.ffn_down())?;
            project_matvec(workspace.proj_hidden, &down_view, workspace.ffn_act)?;
        }
        add_residual_in_place(workspace.hidden, workspace.proj_hidden);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "zero-allocation LM head logits helper borrows pre-allocated scratch slices explicitly"
    )]
    fn write_lm_logits(
        &self,
        weights: &dyn ModelWeights,
        hidden_state: &[f32],
        normed_out: &mut [f32],
        logits_out: &mut [f32],
        repack_cache: Option<&Phi3ModelRepack>,
        scratch_norm_w: &mut [f32],
        scratch_probe: &mut [f32],
        scratch_scope: Option<&ScratchScope<'_>>,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        if logits_out.len() != self.vocab_size {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "logits buffer len {} diverges from model vocab_size {}",
                    logits_out.len(),
                    self.vocab_size
                ),
            });
        }

        if weights.tensor_count() == 0 {
            scratch_norm_w.fill(1.0_f32);
            rmsnorm_f32(normed_out, hidden_state, scratch_norm_w, self.rms_eps)?;
            let norm_sum: f32 = normed_out.iter().sum();
            logits_out.fill(0.0);
            for (v, slot) in logits_out.iter_mut().enumerate().take(self.vocab_size) {
                synthesize_embedding(u32::try_from(v).unwrap_or(0), scratch_probe);
                *slot = dot_product_f32(normed_out, scratch_probe) + norm_sum * 1.0e-3;
            }
            return Ok(());
        }

        let lm_head_view = if let Some(cache) = repack_cache {
            rmsnorm_f32(normed_out, hidden_state, cache.output_norm(), self.rms_eps)?;
            weights.tensor(cache.lm_head_tensor_name())?
        } else {
            let out_norm_view = weights.tensor("output_norm.weight")?;
            decode_1d_norm_into(&out_norm_view, hidden, scratch_norm_w)?;
            rmsnorm_f32(normed_out, hidden_state, scratch_norm_w, self.rms_eps)?;
            if weights.contains_tensor("output.weight") {
                weights.tensor("output.weight")?
            } else {
                weights.tensor("token_embd.weight")?
            }
        };
        let (lm_vocab, _) = resolve_2d_dims(&lm_head_view, hidden)?;
        if lm_vocab != self.vocab_size {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "LM head '{}' row count {lm_vocab} diverges from model vocab_size {}",
                    lm_head_view.name(),
                    self.vocab_size
                ),
            });
        }

        project_lm_head_parallel(logits_out, &lm_head_view, normed_out, scratch_scope)
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

struct BatchPrefillWorkspace {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    proj_hidden: Vec<f32>,
    single_normed: Vec<f32>,
    unit_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_act: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    scores_scratch: Vec<f32>,
    #[cfg(feature = "with-kernels")]
    q_act: QuantizedActQ8,
}

impl BatchPrefillWorkspace {
    fn new(recipe: &Phi3Recipe, m: usize, end_pos: usize) -> Self {
        let hidden = recipe.hidden_dim;
        let total_q = recipe.n_heads_q * recipe.head_dim;
        let total_kv = recipe.n_heads_kv * recipe.head_dim;
        let half_rot = recipe.effective_rotary_dim() / 2;
        let scratch_len = recipe
            .sliding_window
            .map_or(end_pos, |w| w.min(end_pos))
            .max(1);
        Self {
            hidden: vec![0.0_f32; m * hidden],
            normed: vec![0.0_f32; m * hidden],
            proj_hidden: vec![0.0_f32; m * hidden],
            single_normed: vec![0.0_f32; hidden],
            unit_norm: vec![1.0_f32; hidden],
            q: vec![0.0_f32; m * total_q],
            k: vec![0.0_f32; m * total_kv],
            v: vec![0.0_f32; m * total_kv],
            attn_out: vec![0.0_f32; m * total_q],
            gate: vec![0.0_f32; m * recipe.ffn_dim],
            up: vec![0.0_f32; m * recipe.ffn_dim],
            ffn_act: vec![0.0_f32; m * recipe.ffn_dim],
            cos: vec![0.0_f32; half_rot],
            sin: vec![0.0_f32; half_rot],
            scores_scratch: vec![0.0_f32; scratch_len],
            #[cfg(feature = "with-kernels")]
            q_act: QuantizedActQ8::new(),
        }
    }
}

fn append_and_attend_layer_batch(
    recipe: &Phi3Recipe,
    layer: usize,
    m: usize,
    start_pos: usize,
    end_pos: usize,
    kv_view: &mut KvCacheView<'_>,
    ws: &mut BatchPrefillWorkspace,
) -> Result<(), K3Error> {
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let attn_scale = 1.0_f32 / (recipe.head_dim as f32).sqrt();
    let kv_start = start_pos * total_kv;
    let kv_end = end_pos * total_kv;
    let (k_plane, v_plane) = kv_view.layer_kv_mut(layer)?;
    let (Some(k_dst), Some(v_dst)) = (
        k_plane.get_mut(kv_start..kv_end),
        v_plane.get_mut(kv_start..kv_end),
    ) else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: kv_start,
            byte_len: (kv_end - kv_start) * size_of::<f32>(),
            buffer_len: k_plane.len().min(v_plane.len()) * size_of::<f32>(),
            reason: format!(
                "layer {layer} KV slice [{kv_start}..{kv_end}) exceeds plane length ({})",
                k_plane.len()
            ),
        });
    };
    k_dst.copy_from_slice(&ws.k);
    v_dst.copy_from_slice(&ws.v);
    gqa_prefill_causal_f32(
        &mut ws.attn_out,
        &ws.q,
        &k_plane[..kv_end],
        &v_plane[..kv_end],
        m,
        recipe.n_heads_q,
        recipe.n_heads_kv,
        recipe.head_dim,
        end_pos,
        start_pos,
        recipe.sliding_window,
        attn_scale,
        &mut ws.scores_scratch,
    )
}

fn project_repacked_batch_qkv(
    lr: &Phi3LayerRepack,
    m: usize,
    ws: &mut BatchPrefillWorkspace,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let hidden = lr.wq().cols();
        ws.q_act.quantize_into(&ws.normed, m, hidden)?;
        lr.wq().gemm_quantized_parallel(&mut ws.q, &ws.q_act, m)?;
        lr.wk().gemm_quantized_parallel(&mut ws.k, &ws.q_act, m)?;
        lr.wv().gemm_quantized_parallel(&mut ws.v, &ws.q_act, m)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        lr.wq().gemm(&mut ws.q, &ws.normed, m)?;
        lr.wk().gemm(&mut ws.k, &ws.normed, m)?;
        lr.wv().gemm(&mut ws.v, &ws.normed, m)
    }
}

fn project_repacked_batch_o(
    lr: &Phi3LayerRepack,
    m: usize,
    ws: &mut BatchPrefillWorkspace,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let total_q = lr.wo().cols();
        ws.q_act.quantize_into(&ws.attn_out, m, total_q)?;
        lr.wo()
            .gemm_quantized_parallel(&mut ws.proj_hidden, &ws.q_act, m)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        lr.wo().gemm(&mut ws.proj_hidden, &ws.attn_out, m)
    }
}

fn project_repacked_batch_mlp(
    lr: &Phi3LayerRepack,
    m: usize,
    ws: &mut BatchPrefillWorkspace,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let hidden = lr.wgate().cols();
        let ffn_dim = lr.wdown().cols();
        ws.q_act.quantize_into(&ws.normed, m, hidden)?;
        lr.wgate()
            .gemm_quantized_parallel(&mut ws.gate, &ws.q_act, m)?;
        lr.wup().gemm_quantized_parallel(&mut ws.up, &ws.q_act, m)?;
        swiglu_layer_f32(&mut ws.ffn_act, &ws.gate, &ws.up)?;
        ws.q_act.quantize_into(&ws.ffn_act, m, ffn_dim)?;
        lr.wdown()
            .gemm_quantized_parallel(&mut ws.proj_hidden, &ws.q_act, m)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        lr.wgate().gemm(&mut ws.gate, &ws.normed, m)?;
        lr.wup().gemm(&mut ws.up, &ws.normed, m)?;
        swiglu_layer_f32(&mut ws.ffn_act, &ws.gate, &ws.up)?;
        lr.wdown().gemm(&mut ws.proj_hidden, &ws.ffn_act, m)
    }
}

struct DecodeScratch<'a, 'b> {
    hidden: &'a mut [f32],
    normed: &'a mut [f32],
    proj_hidden: &'a mut [f32],
    norm_w: &'a mut [f32],
    q: &'a mut [f32],
    k: &'a mut [f32],
    v: &'a mut [f32],
    attn_out: &'a mut [f32],
    gate: &'a mut [f32],
    up: &'a mut [f32],
    ffn_act: &'a mut [f32],
    cos: &'a mut [f32],
    sin: &'a mut [f32],
    scores: &'a mut [f32],
    #[cfg_attr(
        not(feature = "with-kernels"),
        expect(
            dead_code,
            reason = "only quantized Q8 scratch views consume DecodeScratch::scope"
        )
    )]
    scope: &'a ScratchScope<'b>,
}

impl<'a, 'b> DecodeScratch<'a, 'b> {
    fn lease(
        recipe: &Phi3Recipe,
        seq_len: usize,
        scope: &'a ScratchScope<'b>,
    ) -> Result<Self, K3Error> {
        let hidden = recipe.hidden_dim;
        let total_q = recipe.n_heads_q * recipe.head_dim;
        let total_kv = recipe.n_heads_kv * recipe.head_dim;
        let half_rot = (recipe.effective_rotary_dim() / 2).max(1);
        Ok(Self {
            hidden: scope.alloc_f32(hidden)?,
            normed: scope.alloc_f32(hidden)?,
            proj_hidden: scope.alloc_f32(hidden)?,
            norm_w: scope.alloc_f32(hidden)?,
            q: scope.alloc_f32(total_q)?,
            k: scope.alloc_f32(total_kv)?,
            v: scope.alloc_f32(total_kv)?,
            attn_out: scope.alloc_f32(total_q)?,
            gate: scope.alloc_f32(recipe.ffn_dim)?,
            up: scope.alloc_f32(recipe.ffn_dim)?,
            ffn_act: scope.alloc_f32(recipe.ffn_dim)?,
            cos: scope.alloc_f32(half_rot)?,
            sin: scope.alloc_f32(half_rot)?,
            scores: scope.alloc_f32(seq_len.max(1))?,
            scope,
        })
    }
}

fn project_repacked_qkv(
    lr: &Phi3LayerRepack,
    workspace: &mut DecodeScratch<'_, '_>,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let q_act = QuantizedActQ8View::quantize_row_in_scope(workspace.normed, workspace.scope)?;
        lr.wq().gemv_view_parallel(workspace.q, &q_act)?;
        lr.wk().gemv_view_parallel(workspace.k, &q_act)?;
        lr.wv().gemv_view_parallel(workspace.v, &q_act)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        lr.wq().gemv_parallel(workspace.q, workspace.normed)?;
        lr.wk().gemv_parallel(workspace.k, workspace.normed)?;
        lr.wv().gemv_parallel(workspace.v, workspace.normed)
    }
}

fn project_repacked_mlp_up(
    lr: &Phi3LayerRepack,
    workspace: &mut DecodeScratch<'_, '_>,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let q_act = QuantizedActQ8View::quantize_row_in_scope(workspace.normed, workspace.scope)?;
        lr.wgate().gemv_view_parallel(workspace.gate, &q_act)?;
        lr.wup().gemv_view_parallel(workspace.up, &q_act)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        lr.wgate().gemv_parallel(workspace.gate, workspace.normed)?;
        lr.wup().gemv_parallel(workspace.up, workspace.normed)
    }
}

fn add_residual_in_place(hidden: &mut [f32], delta: &[f32]) {
    for (h, &d) in hidden.iter_mut().zip(delta) {
        *h += d;
    }
}

fn resolve_vocab_size_from_metadata(
    metadata: &ModelMetadata,
    hidden_dim: usize,
    default_vocab: usize,
    arch: &str,
) -> Result<usize, K3Error> {
    let embd_vocab = infer_tensor_vocab_rows(metadata, "token_embd.weight", hidden_dim, arch)?;
    let out_vocab = infer_tensor_vocab_rows(metadata, "output.weight", hidden_dim, arch)?;
    if let (Some(ev), Some(ov)) = (embd_vocab, out_vocab)
        && ev != ov
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "token_embd.weight row count ({ev}) diverges from output.weight row count ({ov})"
            ),
        });
    }
    let tensor_vocab = out_vocab.or(embd_vocab);
    match metadata.vocab_size() {
        Some(0) => Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: "GGUF metadata vocab_size must be positive".to_owned(),
        }),
        Some(meta_vocab) => {
            if let Some(tv) = tensor_vocab
                && tv != meta_vocab
            {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!(
                        "GGUF vocab_size ({meta_vocab}) diverges from model weight row count ({tv})"
                    ),
                });
            }
            Ok(meta_vocab)
        }
        None => Ok(tensor_vocab.unwrap_or(default_vocab)),
    }
}

fn infer_tensor_vocab_rows(
    metadata: &ModelMetadata,
    tensor_name: &str,
    hidden_dim: usize,
    arch: &str,
) -> Result<Option<usize>, K3Error> {
    let Some(t) = metadata.find_tensor(tensor_name) else {
        return Ok(None);
    };
    let s = t.shape();
    if s.len() == 2 && s[0] == hidden_dim && s[1] > 0 {
        Ok(Some(s[1]))
    } else if s.len() == 2 && s[1] == hidden_dim && s[0] > 0 {
        Ok(Some(s[0]))
    } else {
        Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "{tensor_name} shape {s:?} does not align with hidden_dim {hidden_dim}"
            ),
        })
    }
}

fn gguf_optional_token_id(
    gguf: &GgufFile<'_>,
    key: &str,
    vocab_size: usize,
    arch: &str,
) -> Result<Option<u32>, K3Error> {
    let Some(val) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    let id = val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: format!("{key} must be an unsigned integer token ID"),
    })?;
    if usize::try_from(id).is_ok_and(|idx| idx < vocab_size) {
        Ok(Some(id))
    } else {
        Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("{key} ({id}) out of bounds for vocab_size {vocab_size}"),
        })
    }
}

#[derive(Default)]
struct ScannedPhiTokens {
    token_list_len: Option<usize>,
    end_token_id: Option<u32>,
    im_sep_token_id: Option<u32>,
    im_end_token_id: Option<u32>,
    endoftext_token_id: Option<u32>,
}

fn scan_phi_special_tokens(
    gguf: &GgufFile<'_>,
    vocab_size: usize,
    arch: &str,
) -> Result<ScannedPhiTokens, K3Error> {
    let Some(tok_val) = gguf.metadata_value("tokenizer.ggml.tokens") else {
        return Ok(ScannedPhiTokens::default());
    };
    let Some(GgufArray::String(tokens)) = tok_val.as_array() else {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: "tokenizer.ggml.tokens must be a GGUF string array".to_owned(),
        });
    };
    if tokens.len() > vocab_size {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "tokenizer.ggml.tokens length ({}) exceeds vocab_size ({vocab_size})",
                tokens.len()
            ),
        });
    }
    let mut out = ScannedPhiTokens {
        token_list_len: Some(tokens.len()),
        ..ScannedPhiTokens::default()
    };
    for (idx, tok) in tokens.iter().enumerate() {
        if let Ok(id) = u32::try_from(idx) {
            match *tok {
                "<|end|>" => out.end_token_id = Some(id),
                "<|im_sep|>" => out.im_sep_token_id = Some(id),
                "<|im_end|>" => out.im_end_token_id = Some(id),
                "<|endoftext|>" => out.endoftext_token_id = Some(id),
                _ => {}
            }
        }
    }
    Ok(out)
}

fn detect_phi_chat_dialect(
    arch: &str,
    im_sep_token_id: Option<u32>,
    end_token_id: Option<u32>,
    im_end_token_id: Option<u32>,
    chat_template: Option<&str>,
) -> PhiChatDialect {
    if im_sep_token_id.is_some()
        || chat_template.is_some_and(|t| t.contains("<|im_sep|>"))
        || (arch == "phi4" && end_token_id.is_none() && im_end_token_id.is_some())
    {
        PhiChatDialect::Phi4ImSep
    } else {
        PhiChatDialect::Phi3End
    }
}

fn collect_stop_token_ids(
    dialect: PhiChatDialect,
    eos_token_id: Option<u32>,
    end_turn_id: Option<u32>,
    scanned: &ScannedPhiTokens,
) -> Vec<u32> {
    let mut stop_token_ids = Vec::new();
    match dialect {
        PhiChatDialect::Phi3End => {
            push_unique_stop_id(&mut stop_token_ids, scanned.end_token_id);
            push_unique_stop_id(&mut stop_token_ids, eos_token_id);
            push_unique_stop_id(&mut stop_token_ids, end_turn_id);
            push_unique_stop_id(&mut stop_token_ids, scanned.endoftext_token_id);
            push_unique_stop_id(&mut stop_token_ids, scanned.im_end_token_id);
        }
        PhiChatDialect::Phi4ImSep => {
            push_unique_stop_id(&mut stop_token_ids, scanned.im_end_token_id);
            push_unique_stop_id(&mut stop_token_ids, scanned.im_sep_token_id);
            push_unique_stop_id(&mut stop_token_ids, eos_token_id);
            push_unique_stop_id(&mut stop_token_ids, end_turn_id);
            push_unique_stop_id(&mut stop_token_ids, scanned.endoftext_token_id);
            push_unique_stop_id(&mut stop_token_ids, scanned.end_token_id);
        }
    }
    stop_token_ids
}

fn push_unique_stop_id(dst: &mut Vec<u32>, id: Option<u32>) {
    if let Some(tok) = id
        && !dst.contains(&tok)
    {
        dst.push(tok);
    }
}

const fn aligned_f32_scratch_bytes(elems: usize) -> usize {
    let bytes = elems.saturating_mul(size_of::<f32>());
    if bytes == 0 {
        128
    } else {
        bytes.div_ceil(128).saturating_mul(128)
    }
}

impl Phi3Recipe {
    /// Returns the pre-formatted stack-allocated tensor name descriptor for `layer` (`#33`).
    #[must_use]
    pub const fn layer_descriptor(layer: usize) -> Phi3LayerDescriptor {
        Phi3LayerDescriptor::for_layer(layer)
    }

    /// Pre-builds all per-layer [`Phi3LayerDescriptor`] entries for `0..self.num_layers`.
    #[must_use]
    pub fn layer_descriptors(&self) -> Vec<Phi3LayerDescriptor> {
        (0..self.num_layers)
            .map(Phi3LayerDescriptor::for_layer)
            .collect()
    }

    /// Returns the minimum `128-byte`-aligned [`ScratchWorkspace`] capacity (in bytes) required to
    /// execute single-token `decode_step` up to `max_seq_len` cached tokens without heap
    /// allocation (`#33`).
    #[must_use]
    pub const fn required_decode_scratch_bytes(&self, max_seq_len: usize) -> usize {
        let total_q = self.n_heads_q.saturating_mul(self.head_dim);
        let total_kv = self.n_heads_kv.saturating_mul(self.head_dim);
        let rot_dim = match self.rotary_dim {
            Some(rd) if rd > 0 && rd <= self.head_dim => rd,
            _ => self.head_dim,
        };
        let half_rot = rot_dim / 2;
        let half_rot_min = if half_rot == 0 { 1 } else { half_rot };
        let seq_min = if max_seq_len == 0 { 1 } else { max_seq_len };
        #[cfg(feature = "with-kernels")]
        let act_view_headroom = {
            let per_layer_act_view = QuantizedActQ8View::scratch_bytes_for_row(self.hidden_dim)
                .saturating_mul(2)
                .saturating_add(QuantizedActQ8View::scratch_bytes_for_row(total_q))
                .saturating_add(QuantizedActQ8View::scratch_bytes_for_row(self.ffn_dim));
            let lm_q8_headroom = self.hidden_dim.saturating_mul(2).saturating_add(512);
            per_layer_act_view
                .saturating_mul(self.num_layers)
                .saturating_add(lm_q8_headroom)
        };
        #[cfg(not(feature = "with-kernels"))]
        let act_view_headroom = 0_usize;
        aligned_f32_scratch_bytes(self.hidden_dim)
            .saturating_mul(4)
            .saturating_add(aligned_f32_scratch_bytes(total_q).saturating_mul(2))
            .saturating_add(aligned_f32_scratch_bytes(total_kv).saturating_mul(2))
            .saturating_add(aligned_f32_scratch_bytes(self.ffn_dim).saturating_mul(3))
            .saturating_add(aligned_f32_scratch_bytes(half_rot_min).saturating_mul(2))
            .saturating_add(aligned_f32_scratch_bytes(seq_min))
            .saturating_add(act_view_headroom)
    }

    /// Validates runtime geometry and weight tensor shapes/quantizations against `self`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] if geometry,
    /// vocab bounds, or tensor formats fail validation.
    pub(crate) fn validate_weights(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        self.validate_runtime_geometry()?;
        if weights.tensor_count() > 0 {
            let embd_view = weights.tensor("token_embd.weight")?;
            if !self.supports_quant(embd_view.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    self.architecture,
                    embd_view.name(),
                    embd_view.quant(),
                ));
            }
            let (embd_rows, _) = resolve_2d_dims(&embd_view, self.hidden_dim)?;
            if embd_rows != self.vocab_size {
                return Err(K3Error::CapabilityMismatch {
                    architecture: self.architecture.to_owned(),
                    reason: format!(
                        "token_embd.weight row count ({embd_rows}) diverges from model vocab_size ({})",
                        self.vocab_size
                    ),
                });
            }

            let out_norm_view = weights.tensor("output_norm.weight")?;
            if !self.supports_quant(out_norm_view.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    self.architecture,
                    out_norm_view.name(),
                    out_norm_view.quant(),
                ));
            }
            if out_norm_view.num_elements() != self.hidden_dim {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: out_norm_view.name().to_owned(),
                    offset: out_norm_view.offset(),
                    byte_len: out_norm_view.byte_len(),
                    buffer_len: out_norm_view.as_bytes().len(),
                    reason: format!(
                        "output_norm.weight len {} != hidden_dim {}",
                        out_norm_view.num_elements(),
                        self.hidden_dim
                    ),
                });
            }

            if let Ok(out_view) = weights.tensor("output.weight") {
                if !self.supports_quant(out_view.quant()) {
                    return Err(K3Error::unsupported_tensor_quant(
                        self.architecture,
                        out_view.name(),
                        out_view.quant(),
                    ));
                }
                let (out_rows, _) = resolve_2d_dims(&out_view, self.hidden_dim)?;
                if out_rows != self.vocab_size {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: self.architecture.to_owned(),
                        reason: format!(
                            "output.weight row count ({out_rows}) diverges from model vocab_size ({})",
                            self.vocab_size
                        ),
                    });
                }
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

    /// Pre-splits all fused `attn_qkv.weight` and `ffn_up.weight` tensors into independent
    /// `128-byte`-aligned sub-matrices and caches decoded 1D norm vectors once at model setup.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] if model validation
    /// or tensor slicing fails.
    pub fn build_repack_cache(
        &self,
        weights: &dyn ModelWeights,
    ) -> Result<Phi3ModelRepack, K3Error> {
        Phi3ModelRepack::from_weights(self, weights)
    }

    /// Validates vocabulary bounds, tied/untied LM-head alignment, special-token bounds, and
    /// chat-dialect stop tokens (`Phi-3`/`Phi-3.5` `<|end|>` vs `Phi-4` `<|im_sep|>`) from `gguf`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if GGUF `vocab_size`, `tokenizer.ggml.tokens`,
    /// special token IDs, `token_embd.weight`, or `output.weight` shapes diverge.
    pub fn extract_stop_metadata(&self, gguf: &GgufFile<'_>) -> Result<PhiStopMetadata, K3Error> {
        let has_tied_lm_head = self.validate_gguf_vocab_alignment(gguf.metadata())?;
        let bos_token_id = gguf_optional_token_id(
            gguf,
            "tokenizer.ggml.bos_token_id",
            self.vocab_size,
            self.architecture,
        )?;
        let eos_token_id = gguf_optional_token_id(
            gguf,
            "tokenizer.ggml.eos_token_id",
            self.vocab_size,
            self.architecture,
        )?;
        let end_turn_id = gguf_optional_token_id(
            gguf,
            "tokenizer.ggml.eot_token_id",
            self.vocab_size,
            self.architecture,
        )?;
        let _ = gguf_optional_token_id(
            gguf,
            "tokenizer.ggml.padding_token_id",
            self.vocab_size,
            self.architecture,
        )?;

        let scanned = scan_phi_special_tokens(gguf, self.vocab_size, self.architecture)?;
        let chat_template = gguf
            .metadata_value("tokenizer.chat_template")
            .and_then(GgufValue::as_str);
        let dialect = detect_phi_chat_dialect(
            self.architecture,
            scanned.im_sep_token_id,
            scanned.end_token_id,
            scanned.im_end_token_id,
            chat_template,
        );
        let stop_token_ids = collect_stop_token_ids(dialect, eos_token_id, end_turn_id, &scanned);

        Ok(PhiStopMetadata {
            dialect,
            vocab_size: self.vocab_size,
            token_list_len: scanned.token_list_len,
            bos_token_id,
            eos_token_id,
            eot_token_id: end_turn_id,
            end_token_id: scanned.end_token_id,
            im_sep_token_id: scanned.im_sep_token_id,
            im_end_token_id: scanned.im_end_token_id,
            endoftext_token_id: scanned.endoftext_token_id,
            stop_token_ids,
            has_tied_lm_head,
        })
    }

    fn validate_gguf_vocab_alignment(&self, meta: &ModelMetadata) -> Result<bool, K3Error> {
        if let Some(meta_vocab) = meta.vocab_size()
            && meta_vocab != self.vocab_size
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "GGUF vocab_size ({meta_vocab}) diverges from recipe vocab_size ({})",
                    self.vocab_size
                ),
            });
        }

        let embd_vocab = infer_tensor_vocab_rows(
            meta,
            "token_embd.weight",
            self.hidden_dim,
            self.architecture,
        )?;
        if let Some(ev) = embd_vocab
            && ev != self.vocab_size
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "token_embd.weight row count ({ev}) diverges from vocab_size ({})",
                    self.vocab_size
                ),
            });
        }

        let out_vocab =
            infer_tensor_vocab_rows(meta, "output.weight", self.hidden_dim, self.architecture)?;
        if let Some(ov) = out_vocab
            && ov != self.vocab_size
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "output.weight row count ({ov}) diverges from vocab_size ({})",
                    self.vocab_size
                ),
            });
        }

        Ok(out_vocab.is_none())
    }

    fn validate_gguf_contracts(&self, gguf: &GgufFile<'_>) -> Result<(), K3Error> {
        let _ = self.extract_stop_metadata(gguf)?;
        Ok(())
    }

    fn validate_repack_cache(&self, cache: &Phi3ModelRepack) -> Result<(), K3Error> {
        if cache.architecture() != self.architecture
            || cache.hidden_dim() != self.hidden_dim
            || cache.vocab_size() != self.vocab_size
            || cache.layers().len() != self.num_layers
            || cache.output_norm().len() != self.hidden_dim
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "Phi3ModelRepack cache mismatch (cache: arch={}, hidden={}, vocab={}, layers={}; recipe: arch={}, hidden={}, vocab={}, layers={})",
                    cache.architecture(),
                    cache.hidden_dim(),
                    cache.vocab_size(),
                    cache.layers().len(),
                    self.architecture,
                    self.hidden_dim,
                    self.vocab_size,
                    self.num_layers
                ),
            });
        }
        Ok(())
    }

    /// Executes a prefill step across `ctx.prompt_tokens()` using pre-split `RepackedWeight`
    /// sub-matrices and cached 1D norm vectors from `cache`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`], [`K3Error::ContextLengthExceeded`], or
    /// [`K3Error::TensorBoundsError`] on invalid cache, context, or buffer dimensions.
    pub fn prefill_step_with_repack(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &Phi3ModelRepack,
    ) -> Result<(), K3Error> {
        self.validate_repack_cache(cache)?;
        self.prefill_step_inner(ctx, weights, Some(cache))
    }

    /// Executes a single-token decode step using pre-split `RepackedWeight` sub-matrices and
    /// cached 1D norm vectors from `cache`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`], [`K3Error::ContextLengthExceeded`], or
    /// [`K3Error::TensorBoundsError`] on invalid cache, context, or buffer dimensions.
    pub fn decode_step_with_repack(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &Phi3ModelRepack,
    ) -> Result<(), K3Error> {
        self.validate_repack_cache(cache)?;
        self.decode_step_inner(ctx, weights, Some(cache))
    }

    fn embed_prompt_batch(
        &self,
        prompt_tokens: &[u32],
        weights: &dyn ModelWeights,
        hidden_batch: &mut [f32],
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        if weights.tensor_count() == 0 {
            for (t, &token_id) in prompt_tokens.iter().enumerate() {
                synthesize_embedding(token_id, &mut hidden_batch[t * hidden..(t + 1) * hidden]);
            }
            return Ok(());
        }

        let embd_view = weights.tensor("token_embd.weight")?;
        for (t, &token_id) in prompt_tokens.iter().enumerate() {
            dequantize_row_to_f32(
                &embd_view,
                token_id as usize,
                hidden,
                &mut hidden_batch[t * hidden..(t + 1) * hidden],
            )?;
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "batched prefill layer passes bounds and KV view explicitly"
    )]
    fn forward_synthetic_prefill_layer(
        &self,
        layer: usize,
        m: usize,
        start_pos: usize,
        end_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        ws: &mut BatchPrefillWorkspace,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        let total_q = self.n_heads_q * self.head_dim;
        let total_kv = self.n_heads_kv * self.head_dim;
        let ffn_dim = self.ffn_dim;

        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            &ws.unit_norm,
            m,
            hidden,
            self.rms_eps,
        )?;
        for t in 0..m {
            synthesize_qkv(
                layer,
                &ws.normed[t * hidden..(t + 1) * hidden],
                &mut ws.q[t * total_q..(t + 1) * total_q],
                &mut ws.k[t * total_kv..(t + 1) * total_kv],
                &mut ws.v[t * total_kv..(t + 1) * total_kv],
            );
        }
        self.apply_rope_batch(&mut ws.q, &mut ws.k, m, start_pos, &mut ws.cos, &mut ws.sin)?;
        append_and_attend_layer_batch(self, layer, m, start_pos, end_pos, kv_view, ws)?;

        for t in 0..m {
            accumulate_folded_proj(
                &ws.attn_out[t * total_q..(t + 1) * total_q],
                &mut ws.hidden[t * hidden..(t + 1) * hidden],
                0.25,
            );
        }
        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            &ws.unit_norm,
            m,
            hidden,
            self.rms_eps,
        )?;
        for t in 0..m {
            synthesize_gate_up(
                ffn_dim,
                &ws.normed[t * hidden..(t + 1) * hidden],
                &mut ws.gate[t * ffn_dim..(t + 1) * ffn_dim],
                &mut ws.up[t * ffn_dim..(t + 1) * ffn_dim],
            );
        }
        swiglu_layer_f32(&mut ws.ffn_act, &ws.gate, &ws.up)?;
        for t in 0..m {
            accumulate_folded_proj(
                &ws.ffn_act[t * ffn_dim..(t + 1) * ffn_dim],
                &mut ws.hidden[t * hidden..(t + 1) * hidden],
                0.25,
            );
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "batched prefill layer passes repacked weights and bounds explicitly"
    )]
    fn forward_repacked_prefill_layer(
        &self,
        layer: usize,
        lr: &Phi3LayerRepack,
        m: usize,
        start_pos: usize,
        end_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        ws: &mut BatchPrefillWorkspace,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            lr.attn_norm(),
            m,
            hidden,
            self.rms_eps,
        )?;
        project_repacked_batch_qkv(lr, m, ws)?;
        self.apply_rope_batch(&mut ws.q, &mut ws.k, m, start_pos, &mut ws.cos, &mut ws.sin)?;
        append_and_attend_layer_batch(self, layer, m, start_pos, end_pos, kv_view, ws)?;

        project_repacked_batch_o(lr, m, ws)?;
        add_residual_in_place(&mut ws.hidden, &ws.proj_hidden);

        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            lr.ffn_norm(),
            m,
            hidden,
            self.rms_eps,
        )?;
        project_repacked_batch_mlp(lr, m, ws)?;
        add_residual_in_place(&mut ws.hidden, &ws.proj_hidden);
        Ok(())
    }

    #[cfg(not(feature = "with-kernels"))]
    #[allow(
        clippy::too_many_arguments,
        reason = "portable scalar prefill fallback passes weights and bounds explicitly"
    )]
    fn forward_scalar_prefill_layer(
        &self,
        layer: usize,
        weights: &dyn ModelWeights,
        m: usize,
        start_pos: usize,
        end_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        ws: &mut BatchPrefillWorkspace,
    ) -> Result<(), K3Error> {
        let hidden = self.hidden_dim;
        let total_q = self.n_heads_q * self.head_dim;
        let total_kv = self.n_heads_kv * self.head_dim;
        let ffn_dim = self.ffn_dim;

        let desc = Phi3LayerDescriptor::for_layer(layer);
        let attn_norm_view = weights.tensor(desc.attn_norm())?;
        decode_1d_norm_into(&attn_norm_view, hidden, &mut ws.single_normed)?;
        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            &ws.single_normed,
            m,
            hidden,
            self.rms_eps,
        )?;
        for t in 0..m {
            project_layer_qkv(
                &desc,
                hidden,
                total_q,
                total_kv,
                weights,
                &ws.normed[t * hidden..(t + 1) * hidden],
                &mut ws.q[t * total_q..(t + 1) * total_q],
                &mut ws.k[t * total_kv..(t + 1) * total_kv],
                &mut ws.v[t * total_kv..(t + 1) * total_kv],
            )?;
        }

        self.apply_rope_batch(&mut ws.q, &mut ws.k, m, start_pos, &mut ws.cos, &mut ws.sin)?;
        append_and_attend_layer_batch(self, layer, m, start_pos, end_pos, kv_view, ws)?;

        let o_view = weights.tensor(desc.attn_output())?;
        for t in 0..m {
            project_matvec(
                &mut ws.proj_hidden[t * hidden..(t + 1) * hidden],
                &o_view,
                &ws.attn_out[t * total_q..(t + 1) * total_q],
            )?;
        }
        add_residual_in_place(&mut ws.hidden, &ws.proj_hidden);

        load_ffn_norm_into(&desc, hidden, weights, &mut ws.single_normed)?;
        rmsnorm_batch_f32(
            &mut ws.normed,
            &ws.hidden,
            &ws.single_normed,
            m,
            hidden,
            self.rms_eps,
        )?;
        for t in 0..m {
            project_layer_mlp_up(
                &desc,
                hidden,
                ffn_dim,
                weights,
                &ws.normed[t * hidden..(t + 1) * hidden],
                &mut ws.gate[t * ffn_dim..(t + 1) * ffn_dim],
                &mut ws.up[t * ffn_dim..(t + 1) * ffn_dim],
            )?;
        }
        swiglu_layer_f32(&mut ws.ffn_act, &ws.gate, &ws.up)?;

        let down_view = weights.tensor(desc.ffn_down())?;
        for t in 0..m {
            project_matvec(
                &mut ws.proj_hidden[t * hidden..(t + 1) * hidden],
                &down_view,
                &ws.ffn_act[t * ffn_dim..(t + 1) * ffn_dim],
            )?;
        }
        add_residual_in_place(&mut ws.hidden, &ws.proj_hidden);
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "batched layer loop keeps synthetic, cached, and auto-repack dispatch cohesive"
    )]
    fn forward_prefill_layers(
        &self,
        synthetic_mode: bool,
        weights: &dyn ModelWeights,
        repack_cache: Option<&Phi3ModelRepack>,
        m: usize,
        start_pos: usize,
        end_pos: usize,
        kv_view: &mut KvCacheView<'_>,
        ws: &mut BatchPrefillWorkspace,
    ) -> Result<(), K3Error> {
        for layer in 0..self.num_layers {
            if synthetic_mode {
                self.forward_synthetic_prefill_layer(layer, m, start_pos, end_pos, kv_view, ws)?;
            } else if let Some(cache) = repack_cache {
                let lr = cache
                    .layer(layer)
                    .ok_or_else(|| K3Error::CapabilityMismatch {
                        architecture: self.architecture.to_owned(),
                        reason: format!("repack cache missing layer {layer}"),
                    })?;
                self.forward_repacked_prefill_layer(layer, lr, m, start_pos, end_pos, kv_view, ws)?;
            } else {
                #[cfg(feature = "with-kernels")]
                {
                    let lr = get_or_repack_layer(self, layer, weights)?;
                    self.forward_repacked_prefill_layer(
                        layer, &lr, m, start_pos, end_pos, kv_view, ws,
                    )?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let _ = self;
                    self.forward_scalar_prefill_layer(
                        layer, weights, m, start_pos, end_pos, kv_view, ws,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn prefill_step_inner(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        repack_cache: Option<&Phi3ModelRepack>,
    ) -> Result<(), K3Error> {
        self.validate_runtime_geometry()?;
        if ctx.prompt_tokens().is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "prefill_step requires at least one prompt token".to_owned(),
            });
        }

        let prompt_tokens: Vec<u32> = ctx.prompt_tokens().to_vec();
        let m = prompt_tokens.len();
        let start_pos = ctx.position();
        let end_pos = start_pos
            .checked_add(m)
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

        let synthetic_mode = weights.tensor_count() == 0;
        if synthetic_mode && repack_cache.is_some() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "repack cache cannot be used with empty synthetic weights".to_owned(),
            });
        }

        let compute_logits = ctx.compute_logits();
        let geom = self.resolve_kv_geometry(ctx)?;
        if end_pos > geom.seq_capacity {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: geom.seq_capacity,
            });
        }
        let mut ws = BatchPrefillWorkspace::new(self, m, end_pos);
        self.embed_prompt_batch(&prompt_tokens, weights, &mut ws.hidden)?;

        {
            let (logits_buf, kv_buf) = ctx.buffers_mut();
            let mut kv_view = KvCacheView::new(kv_buf, geom)?;
            self.forward_prefill_layers(
                synthetic_mode,
                weights,
                repack_cache,
                m,
                start_pos,
                end_pos,
                &mut kv_view,
                &mut ws,
            )?;

            if compute_logits {
                let last_offset = (m - 1) * self.hidden_dim;
                self.write_lm_logits(
                    weights,
                    &ws.hidden[last_offset..last_offset + self.hidden_dim],
                    &mut ws.single_normed,
                    logits_buf,
                    repack_cache,
                    &mut ws.normed[..self.hidden_dim],
                    &mut ws.proj_hidden[..self.hidden_dim],
                    None,
                )?;
            }
        }
        ctx.advance_position(m)?;
        Ok(())
    }

    fn decode_step_inner(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        repack_cache: Option<&Phi3ModelRepack>,
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
        let required_scratch = self.required_decode_scratch_bytes(abs_pos + 1);
        {
            let (logits_buf, kv_buf, maybe_scratch) = ctx.buffers_and_scratch_mut();
            let mut kv_view = KvCacheView::new(kv_buf, geom)?;
            if let Some(scratch) = maybe_scratch {
                self.decode_step_with_scratch_inner(
                    token_id,
                    abs_pos,
                    weights,
                    repack_cache,
                    &mut kv_view,
                    logits_buf,
                    scratch,
                )?;
            } else {
                let mut fallback = ScratchWorkspace::new(required_scratch)?;
                self.decode_step_with_scratch_inner(
                    token_id,
                    abs_pos,
                    weights,
                    repack_cache,
                    &mut kv_view,
                    logits_buf,
                    &mut fallback,
                )?;
            }
        }
        ctx.advance_position(1)?;
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "single shared scratch decode path for caller-supplied and transient arenas"
    )]
    fn decode_step_with_scratch_inner(
        &self,
        token_id: u32,
        abs_pos: usize,
        weights: &dyn ModelWeights,
        repack_cache: Option<&Phi3ModelRepack>,
        kv_view: &mut KvCacheView<'_>,
        logits_buf: &mut [f32],
        scratch: &mut ScratchWorkspace,
    ) -> Result<(), K3Error> {
        let scope = scratch.scope();
        let mut workspace = DecodeScratch::lease(self, abs_pos + 1, &scope)?;
        self.forward_token(
            token_id,
            abs_pos,
            weights,
            kv_view,
            &mut workspace,
            repack_cache,
        )?;
        self.write_lm_logits(
            weights,
            workspace.hidden,
            workspace.normed,
            logits_buf,
            repack_cache,
            workspace.norm_w,
            workspace.proj_hidden,
            Some(&scope),
        )
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

#[cfg(not(feature = "with-kernels"))]
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
    with_2d_rows_view(tensor, row_start, row_count, expected_cols, |sliced| {
        project_matvec(out, sliced, input)
    })
}

#[cfg(feature = "with-kernels")]
fn project_row_range_shared_q8(
    out: &mut [f32],
    tensor: &TensorView<'_>,
    expected_cols: usize,
    row_start: usize,
    row_count: usize,
    shared: &mut SharedQ8Input<'_>,
) -> Result<(), K3Error> {
    if out.len() != row_count {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "row-slice projection buffer mismatch (out={}, row_count={row_count})",
                out.len()
            ),
        });
    }
    with_2d_rows_view(tensor, row_start, row_count, expected_cols, |sliced| {
        project_matvec_shared_q8(out, sliced, shared)
    })
}

#[expect(clippy::too_many_arguments, reason = "explicit QKV slice projections")]
fn project_layer_qkv(
    desc: &Phi3LayerDescriptor,
    hidden: usize,
    total_q: usize,
    total_kv: usize,
    weights: &dyn ModelWeights,
    normed: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<(), K3Error> {
    if weights.contains_tensor(desc.attn_qkv()) {
        let qkv_view = weights.tensor(desc.attn_qkv())?;
        let (rows, _) = resolve_2d_dims(&qkv_view, hidden)?;
        let expected_rows = total_q + 2 * total_kv;
        if rows != expected_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: desc.attn_qkv().to_owned(),
                offset: qkv_view.offset(),
                byte_len: qkv_view.byte_len(),
                buffer_len: qkv_view.as_bytes().len(),
                reason: format!(
                    "fused attn_qkv rows {rows} != expected total_q + 2*total_kv ({expected_rows})"
                ),
            });
        }
        #[cfg(feature = "with-kernels")]
        {
            let mut shared = SharedQ8Input::new(normed);
            project_row_range_shared_q8(q_out, &qkv_view, hidden, 0, total_q, &mut shared)?;
            project_row_range_shared_q8(k_out, &qkv_view, hidden, total_q, total_kv, &mut shared)?;
            project_row_range_shared_q8(
                v_out,
                &qkv_view,
                hidden,
                total_q + total_kv,
                total_kv,
                &mut shared,
            )
        }
        #[cfg(not(feature = "with-kernels"))]
        {
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
    } else {
        let q_view = weights.tensor(desc.attn_q())?;
        let k_view = weights.tensor(desc.attn_k())?;
        let v_view = weights.tensor(desc.attn_v())?;
        #[cfg(feature = "with-kernels")]
        {
            let mut shared = SharedQ8Input::new(normed);
            project_matvec_shared_q8(q_out, &q_view, &mut shared)?;
            project_matvec_shared_q8(k_out, &k_view, &mut shared)?;
            project_matvec_shared_q8(v_out, &v_view, &mut shared)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            project_matvec(q_out, &q_view, normed)?;
            project_matvec(k_out, &k_view, normed)?;
            project_matvec(v_out, &v_view, normed)
        }
    }
}

fn project_layer_mlp_up(
    desc: &Phi3LayerDescriptor,
    hidden: usize,
    ffn_dim: usize,
    weights: &dyn ModelWeights,
    normed: &[f32],
    gate_out: &mut [f32],
    up_out: &mut [f32],
) -> Result<(), K3Error> {
    let up_view = weights.tensor(desc.ffn_up())?;
    let (rows, _) = resolve_2d_dims(&up_view, hidden)?;
    if rows == 2 * ffn_dim {
        #[cfg(feature = "with-kernels")]
        {
            let mut shared = SharedQ8Input::new(normed);
            project_row_range_shared_q8(gate_out, &up_view, hidden, 0, ffn_dim, &mut shared)?;
            project_row_range_shared_q8(up_out, &up_view, hidden, ffn_dim, ffn_dim, &mut shared)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            project_row_range(gate_out, &up_view, hidden, 0, ffn_dim, normed)?;
            project_row_range(up_out, &up_view, hidden, ffn_dim, ffn_dim, normed)
        }
    } else if rows == ffn_dim {
        let gate_view = weights.tensor(desc.ffn_gate())?;
        #[cfg(feature = "with-kernels")]
        {
            let mut shared = SharedQ8Input::new(normed);
            project_matvec_shared_q8(gate_out, &gate_view, &mut shared)?;
            project_matvec_shared_q8(up_out, &up_view, &mut shared)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            project_matvec(gate_out, &gate_view, normed)?;
            project_matvec(up_out, &up_view, normed)
        }
    } else {
        Err(K3Error::TensorBoundsError {
            tensor_name: desc.ffn_up().to_owned(),
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

fn load_ffn_norm_into(
    desc: &Phi3LayerDescriptor,
    hidden: usize,
    weights: &dyn ModelWeights,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let view = if weights.contains_tensor(desc.ffn_norm()) {
        weights.tensor(desc.ffn_norm())?
    } else {
        weights.tensor(desc.post_attention_norm())?
    };
    decode_1d_norm_into(&view, hidden, out)
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

        self.validate_weights(weights)?;

        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.prefill_step_inner(ctx, weights, None)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.decode_step_inner(ctx, weights, None)
    }
}
