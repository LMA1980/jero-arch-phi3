//! Load-time pre-splitting of fused Phi-3 / Phi-4 weight tensors into independent `128-byte`-aligned
//! `RepackedWeight` sub-matrices and cached 1D `RMSNorm` parameter vectors.
//!
//! Fused `blk.{l}.attn_qkv.weight` (`Q || K || V`) and `blk.{l}.ffn_up.weight` (`Gate || Up`)
//! tensors cannot be repacked monolithically into `8x16` hardware tiles when GQA head geometry or
//! intermediate dimensions place `K`, `V`, or `Up` rows at non-multiple-of-8 offsets. Pre-splitting
//! row-contiguous byte slices before calling `jero_kernels_rvv::RepackedWeight::from_tensor` ensures
//! every logical projection owns dedicated tiles starting cleanly at tile `0`.

#[cfg(feature = "with-kernels")]
use std::collections::HashMap;
#[cfg(feature = "with-kernels")]
use std::ops::Range;
#[cfg(feature = "with-kernels")]
use std::sync::{Arc, OnceLock, RwLock};

use jero_core::{ArchRecipe, K3Error, ModelWeights, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{QuantizedActQ8, RepackedWeight, gemm_repacked, gemm_repacked_rows};

use crate::quant::{decode_1d_norm, quant_row_bytes, resolve_2d_dims};
#[cfg(not(feature = "with-kernels"))]
use crate::quant::{dequantize_row_to_f32, project_matvec};
use crate::recipe::Phi3Recipe;

/// Required byte alignment of packed weight payloads on target hardware tiles.
pub const REPACK_TILE_ALIGNMENT_BYTES: usize = 128;

/// Row count packed per hardware `8x16` matrix tile.
pub const REPACK_TILE_ROWS: usize = 8;

/// Independent row-sliced matrix backed by a `128-byte`-aligned `RepackedWeight` (when
/// `with-kernels` is enabled) or a row-contiguous portable fallback buffer (under
/// `--no-default-features`).
#[derive(Debug, Clone)]
pub struct Phi3SplitMatrix {
    name: String,
    #[cfg(feature = "with-kernels")]
    repacked: RepackedWeight,
    #[cfg(not(feature = "with-kernels"))]
    quant: QuantType,
    #[cfg(not(feature = "with-kernels"))]
    rows: usize,
    #[cfg(not(feature = "with-kernels"))]
    cols: usize,
    #[cfg(not(feature = "with-kernels"))]
    data: Vec<u8>,
}

impl Phi3SplitMatrix {
    /// Slices `row_start..row_start + row_count` from a 2D `tensor` (`[cols, total_rows]` or
    /// `[total_rows, cols]`) and packs the resulting sub-matrix independently so it shares zero
    /// `8x16` tiles with neighboring segments.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on invalid shapes, out-of-range slices, or truncated
    /// buffers, or [`K3Error::CapabilityMismatch`] if the quantization format is unsupported.
    pub fn from_tensor_slice(
        tensor: &TensorView<'_>,
        sub_name: impl Into<String>,
        row_start: usize,
        row_count: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        let name = sub_name.into();
        if row_count == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name,
                offset: row_start,
                byte_len: 0,
                buffer_len: tensor.as_bytes().len(),
                reason: "split matrix slice row_count must be positive".to_owned(),
            });
        }

        let (total_rows, cols) = resolve_2d_dims(tensor, expected_cols)?;
        let row_end =
            row_start
                .checked_add(row_count)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.clone(),
                    offset: row_start,
                    byte_len: row_count,
                    buffer_len: total_rows,
                    reason: "split matrix row slice overflow".to_owned(),
                })?;
        if row_end > total_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name,
                offset: row_start,
                byte_len: row_count,
                buffer_len: total_rows,
                reason: format!(
                    "row slice [{row_start}..{row_end}) exceeds '{}' row count {total_rows}",
                    tensor.name()
                ),
            });
        }

        if tensor.shape().first().copied() == Some(cols) {
            let sub_view = tensor.slice_rows(row_start, row_count)?;
            return Self::pack_subview(&sub_view, name, row_count, cols);
        }

        let row_bytes = quant_row_bytes(cols, tensor.quant()).ok_or_else(|| {
            K3Error::unsupported_tensor_quant("phi3", tensor.name(), tensor.quant())
        })?;
        let byte_start = row_start * row_bytes;
        let byte_end = row_end * row_bytes;
        let sub_shape = [cols, row_count];
        let view_name = tensor.name();
        let sub_view = TensorView::from_bytes(
            view_name,
            tensor.quant(),
            &sub_shape,
            0,
            &tensor.as_bytes()[byte_start..byte_end],
        )?;
        Self::pack_subview(&sub_view, name, row_count, cols)
    }

    /// Validates that a standalone 2D `tensor` has exact dimensions `(expected_rows, expected_cols)`
    /// and packs it into an independent [`Phi3SplitMatrix`].
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if dimensions or buffer length mismatch, or
    /// [`K3Error::CapabilityMismatch`] if the quantization format is unsupported.
    pub fn from_tensor(
        tensor: &TensorView<'_>,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        let (rows, cols) = resolve_2d_dims(tensor, expected_cols)?;
        if rows != expected_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: tensor.byte_len(),
                buffer_len: tensor.as_bytes().len(),
                reason: format!(
                    "expected {} rows for '{}', got {rows}",
                    expected_rows,
                    tensor.name()
                ),
            });
        }
        Self::from_tensor_slice(tensor, tensor.name(), 0, rows, cols)
    }

    fn pack_subview(
        sub_view: &TensorView<'_>,
        name: String,
        rows: usize,
        cols: usize,
    ) -> Result<Self, K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            let repacked = RepackedWeight::from_tensor(sub_view, rows, cols)?;
            Ok(Self { name, repacked })
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            if quant_row_bytes(cols, sub_view.quant()).is_none() {
                return Err(K3Error::unsupported_tensor_quant(
                    "phi3",
                    sub_view.name(),
                    sub_view.quant(),
                ));
            }
            Ok(Self {
                name,
                quant: sub_view.quant(),
                rows,
                cols,
                data: sub_view.as_bytes().to_vec(),
            })
        }
    }

    /// Returns the logical tensor name of this pre-split sub-matrix.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of output rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.rows()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.rows
        }
    }

    /// Returns the number of input columns (`in_features`).
    #[must_use]
    pub fn cols(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.cols()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.cols
        }
    }

    /// Returns the source [`QuantType`] of this sub-matrix.
    #[must_use]
    pub fn quant(&self) -> QuantType {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.quant()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.quant
        }
    }

    /// Returns the number of `8`-row hardware tiles (`rows.div_ceil(8)`) for block-quantized
    /// tensors, or `0` for unquantized dense (`F32` / `F16` / `BF16`) matrices.
    #[must_use]
    pub fn n_tiles(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.n_tiles()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            match self.quant {
                QuantType::F32 | QuantType::F16 | QuantType::BF16 => 0,
                _ => self.rows.div_ceil(REPACK_TILE_ROWS),
            }
        }
    }

    /// Returns the packed backing payload size in bytes.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.aligned_bytes()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.data.len()
        }
    }

    /// Returns the packed payload alignment modulus check (`true` when `128`-byte aligned under
    /// `with-kernels`, or always `true` for the portable scalar fallback).
    #[must_use]
    pub fn is_128_byte_aligned(&self) -> bool {
        #[cfg(feature = "with-kernels")]
        {
            (self.repacked.payload_ptr() as usize) % REPACK_TILE_ALIGNMENT_BYTES == 0
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            true
        }
    }

    /// Returns the underlying `jero_kernels_rvv::RepackedWeight` when `with-kernels` is enabled.
    #[cfg(feature = "with-kernels")]
    #[must_use]
    pub const fn repacked(&self) -> &RepackedWeight {
        &self.repacked
    }

    /// Dequantizes logical row `row` (`0..rows()`) into `out` (`cols()` elements).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `row >= rows()` or `out.len() != cols()`, or
    /// [`K3Error::CapabilityMismatch`] on unsupported scalar formats.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            self.repacked.dequantize_row(row, out)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let shape = [self.cols, self.rows];
            let view = TensorView::from_bytes(&self.name, self.quant, &shape, 0, &self.data)?;
            dequantize_row_to_f32(&view, row, self.cols, out)
        }
    }

    /// Computes matrix-vector product `out = self * input` using dynamically quantized `Q8`
    /// activations when `with-kernels` is enabled, or portable scalar math under
    /// `--no-default-features`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on buffer length mismatches or
    /// [`K3Error::CapabilityMismatch`] on unsupported quantization formats.
    pub fn gemv(&self, out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            let q_act = QuantizedActQ8::quantize_row(input)?;
            self.repacked.gemv(out, &q_act)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let shape = [self.cols, self.rows];
            let view = TensorView::from_bytes(&self.name, self.quant, &shape, 0, &self.data)?;
            project_matvec(out, &view, input)
        }
    }

    /// Computes matrix-vector product `out = self * q_act` reusing an already-quantized input
    /// activation buffer (`QuantizedActQ8`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    #[cfg(feature = "with-kernels")]
    pub fn gemv_quantized(&self, out: &mut [f32], q_act: &QuantizedActQ8) -> Result<(), K3Error> {
        self.repacked.gemv(out, q_act)
    }

    /// Computes batched multi-token GEMM (`[M, cols] x [rows, cols]^T -> [M, rows]`) from raw
    /// `f32` activations `input` using `A100` `XSMTVDotII` / `RVV` `RepackedWeight` tiles when
    /// `with-kernels` is enabled, or portable scalar math under `--no-default-features`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on `m == 0` or buffer dimension mismatches, or
    /// [`K3Error::CapabilityMismatch`] on unsupported quantization formats.
    pub fn gemm(&self, out: &mut [f32], input: &[f32], m: usize) -> Result<(), K3Error> {
        let rows = self.rows();
        let cols = self.cols();
        let expected_in = m.checked_mul(cols);
        let expected_out = m.checked_mul(rows);
        if m == 0 || expected_in != Some(input.len()) || expected_out != Some(out.len()) {
            return Err(K3Error::TensorBoundsError {
                tensor_name: self.name.clone(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(input),
                reason: format!(
                    "batched GEMM shape mismatch (m={m}, rows={rows}, cols={cols}, in={}, out={})",
                    input.len(),
                    out.len()
                ),
            });
        }
        #[cfg(feature = "with-kernels")]
        {
            let mut q_act = QuantizedActQ8::new();
            q_act.quantize_into(input, m, cols)?;
            self.gemm_quantized_parallel(out, &q_act, m)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let shape = [cols, rows];
            let view = TensorView::from_bytes(&self.name, self.quant, &shape, 0, &self.data)?;
            #[cfg(not(miri))]
            if m >= 2 && rayon::current_num_threads() > 1 {
                use rayon::prelude::*;
                return out.par_chunks_exact_mut(rows).enumerate().try_for_each(
                    |(m_idx, out_row)| {
                        let in_row = &input[m_idx * cols..(m_idx + 1) * cols];
                        project_matvec(out_row, &view, in_row)
                    },
                );
            }
            for m_idx in 0..m {
                let in_row = &input[m_idx * cols..(m_idx + 1) * cols];
                let out_row = &mut out[m_idx * rows..(m_idx + 1) * rows];
                project_matvec(out_row, &view, in_row)?;
            }
            Ok(())
        }
    }

    /// Computes full-matrix batched GEMM (`[M, cols] x [rows, cols]^T -> [M, rows]`) reusing an
    /// already-quantized `[M, cols]` activation batch (`QuantizedActQ8`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    #[cfg(feature = "with-kernels")]
    pub fn gemm_quantized(&self, out: &mut [f32], q_act: &QuantizedActQ8) -> Result<(), K3Error> {
        gemm_repacked(out, &self.repacked, q_act)
    }

    /// Computes row-sliced batched GEMM (`[M, cols] x [row_range, cols]^T -> [M, row_range.len()]`)
    /// reusing an already-quantized `[M, cols]` activation batch (`QuantizedActQ8`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `row_range` is out of bounds or dimensions mismatch.
    #[cfg(feature = "with-kernels")]
    pub fn gemm_quantized_rows(
        &self,
        out: &mut [f32],
        q_act: &QuantizedActQ8,
        row_range: Range<usize>,
    ) -> Result<(), K3Error> {
        gemm_repacked_rows(out, &self.repacked, q_act, row_range)
    }

    /// Dispatches row-sliced `gemm_repacked_rows` across active `WorkerPool` threads on
    /// `REPACK_TILE_ROWS`-aligned partitions when inside a parallel Rayon pool (`not(miri)`), or
    /// sequentially otherwise.
    #[cfg(feature = "with-kernels")]
    pub(crate) fn gemm_quantized_parallel(
        &self,
        out: &mut [f32],
        q_act: &QuantizedActQ8,
        m: usize,
    ) -> Result<(), K3Error> {
        #[cfg(not(miri))]
        {
            let rows = self.rows();
            let workers = rayon::current_num_threads();
            if workers > 1 && rows > REPACK_TILE_ROWS && m.saturating_mul(rows) >= 256 {
                let ranges = partition_tile_row_ranges(rows, workers);
                if ranges.len() > 1 {
                    use rayon::prelude::*;
                    let partials: Vec<(Range<usize>, Vec<f32>)> = ranges
                        .into_par_iter()
                        .map(|range| {
                            let sub_n = range.len();
                            let mut sub_out = vec![0.0_f32; m * sub_n];
                            self.gemm_quantized_rows(&mut sub_out, q_act, range.clone())?;
                            Ok((range, sub_out))
                        })
                        .collect::<Result<Vec<_>, K3Error>>()?;
                    for (range, sub_out) in partials {
                        let sub_n = range.len();
                        for m_idx in 0..m {
                            let dst =
                                &mut out[m_idx * rows + range.start..m_idx * rows + range.end];
                            let src = &sub_out[m_idx * sub_n..(m_idx + 1) * sub_n];
                            dst.copy_from_slice(src);
                        }
                    }
                    return Ok(());
                }
            }
        }
        let _ = m;
        self.gemm_quantized(out, q_act)
    }
}

#[cfg(all(feature = "with-kernels", not(miri)))]
fn partition_tile_row_ranges(total_rows: usize, workers: usize) -> Vec<Range<usize>> {
    let tiles = total_rows.div_ceil(REPACK_TILE_ROWS);
    let active_workers = workers.min(tiles).max(1);
    let base_tiles = tiles / active_workers;
    let rem_tiles = tiles % active_workers;
    let mut ranges = Vec::with_capacity(active_workers);
    let mut start = 0_usize;
    for w in 0..active_workers {
        let tile_span = base_tiles + usize::from(w < rem_tiles);
        let end = (start + tile_span * REPACK_TILE_ROWS).min(total_rows);
        if start < end {
            ranges.push(start..end);
        }
        start = end;
        if start >= total_rows {
            break;
        }
    }
    ranges
}

/// Pre-split attention and MLP sub-matrices plus cached 1D `RMSNorm` vectors for a single layer.
#[derive(Debug, Clone)]
pub struct Phi3LayerRepack {
    layer_idx: usize,
    wq: Phi3SplitMatrix,
    wk: Phi3SplitMatrix,
    wv: Phi3SplitMatrix,
    wo: Phi3SplitMatrix,
    wgate: Phi3SplitMatrix,
    wup: Phi3SplitMatrix,
    wdown: Phi3SplitMatrix,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
}

impl Phi3LayerRepack {
    /// Constructs a [`Phi3LayerRepack`] for physical layer `layer` from `weights`, pre-splitting
    /// fused `attn_qkv.weight` (`Q || K || V`) and `ffn_up.weight` (`Gate || Up`) (or separate
    /// unfused tensors when present) and decoding `attn_norm` / `ffn_norm` once.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `layer >= recipe.num_layers()` or any tensor uses
    /// an unsupported quantization format, or [`K3Error::TensorBoundsError`] if any required tensor
    /// is missing or fails shape/slice bounds validation.
    pub fn from_weights(
        recipe: &Phi3Recipe,
        layer: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Self, K3Error> {
        if layer >= recipe.num_layers {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture.to_owned(),
                reason: format!(
                    "layer index {layer} out of bounds for {} layers",
                    recipe.num_layers
                ),
            });
        }

        let hidden = recipe.hidden_dim;
        let q_dim = recipe.n_heads_q * recipe.head_dim;
        let kv_dim = recipe.n_heads_kv * recipe.head_dim;
        let ffn_dim = recipe.ffn_dim;

        let (wq, wk, wv) = split_layer_qkv(recipe, layer, weights, hidden, q_dim, kv_dim)?;
        let wo_view =
            require_supported_tensor(recipe, weights, &format!("blk.{layer}.attn_output.weight"))?;
        let wo = Phi3SplitMatrix::from_tensor(&wo_view, hidden, q_dim)?;

        let (wgate, wup) = split_layer_mlp_up(recipe, layer, weights, hidden, ffn_dim)?;
        let wdown_view =
            require_supported_tensor(recipe, weights, &format!("blk.{layer}.ffn_down.weight"))?;
        let wdown = Phi3SplitMatrix::from_tensor(&wdown_view, hidden, ffn_dim)?;

        let attn_norm_view = weights.tensor(&format!("blk.{layer}.attn_norm.weight"))?;
        let attn_norm = decode_1d_norm(&attn_norm_view, hidden)?;
        let ffn_norm = if let Ok(v) = weights.tensor(&format!("blk.{layer}.ffn_norm.weight")) {
            decode_1d_norm(&v, hidden)?
        } else if let Ok(v) = weights.tensor(&format!("blk.{layer}.post_attention_norm.weight")) {
            decode_1d_norm(&v, hidden)?
        } else {
            vec![1.0_f32; hidden]
        };

        Ok(Self {
            layer_idx: layer,
            wq,
            wk,
            wv,
            wo,
            wgate,
            wup,
            wdown,
            attn_norm,
            ffn_norm,
        })
    }

    /// Returns the 0-based physical layer index.
    #[must_use]
    pub const fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Returns the pre-split query projection matrix (`[q_dim, hidden_dim]`).
    #[must_use]
    pub const fn wq(&self) -> &Phi3SplitMatrix {
        &self.wq
    }

    /// Returns the pre-split key projection matrix (`[kv_dim, hidden_dim]`).
    #[must_use]
    pub const fn wk(&self) -> &Phi3SplitMatrix {
        &self.wk
    }

    /// Returns the pre-split value projection matrix (`[kv_dim, hidden_dim]`).
    #[must_use]
    pub const fn wv(&self) -> &Phi3SplitMatrix {
        &self.wv
    }

    /// Returns the attention output projection matrix (`[hidden_dim, q_dim]`).
    #[must_use]
    pub const fn wo(&self) -> &Phi3SplitMatrix {
        &self.wo
    }

    /// Returns the pre-split MLP gate projection matrix (`[ffn_dim, hidden_dim]`).
    #[must_use]
    pub const fn wgate(&self) -> &Phi3SplitMatrix {
        &self.wgate
    }

    /// Returns the pre-split MLP up projection matrix (`[ffn_dim, hidden_dim]`).
    #[must_use]
    pub const fn wup(&self) -> &Phi3SplitMatrix {
        &self.wup
    }

    /// Returns the MLP down projection matrix (`[hidden_dim, ffn_dim]`).
    #[must_use]
    pub const fn wdown(&self) -> &Phi3SplitMatrix {
        &self.wdown
    }

    /// Returns the decoded `f32` pre-attention `RMSNorm` vector (`hidden_dim`).
    #[must_use]
    pub fn attn_norm(&self) -> &[f32] {
        &self.attn_norm
    }

    /// Returns the decoded `f32` pre-FFN `RMSNorm` vector (`hidden_dim`).
    #[must_use]
    pub fn ffn_norm(&self) -> &[f32] {
        &self.ffn_norm
    }
}

/// Model-scoped pre-split `RepackedWeight` cache covering all layers and decoded 1D norm vectors.
#[derive(Debug, Clone)]
pub struct Phi3ModelRepack {
    architecture: String,
    hidden_dim: usize,
    vocab_size: usize,
    layers: Vec<Phi3LayerRepack>,
    output_norm: Vec<f32>,
}

/// Type alias matching the canonical issue nomenclature (`Phi3RepackCache`).
pub type Phi3RepackCache = Phi3ModelRepack;

impl Phi3ModelRepack {
    /// Pre-splits all fused `attn_qkv.weight` and `ffn_up.weight` projections across all layers
    /// of `recipe` and caches decoded 1D `RMSNorm` vectors once at model setup.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] if model validation,
    /// weight quantization admission, tensor slicing, or norm decoding fails.
    pub fn from_weights(recipe: &Phi3Recipe, weights: &dyn ModelWeights) -> Result<Self, K3Error> {
        recipe.validate_weights(weights)?;
        if weights.tensor_count() == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture.to_owned(),
                reason: "cannot build Phi3ModelRepack cache from empty synthetic ModelWeights"
                    .to_owned(),
            });
        }

        let mut layers = Vec::with_capacity(recipe.num_layers);
        for layer in 0..recipe.num_layers {
            layers.push(Phi3LayerRepack::from_weights(recipe, layer, weights)?);
        }

        let output_norm_view = weights.tensor("output_norm.weight")?;
        let output_norm = decode_1d_norm(&output_norm_view, recipe.hidden_dim)?;

        Ok(Self {
            architecture: recipe.architecture.to_owned(),
            hidden_dim: recipe.hidden_dim,
            vocab_size: recipe.vocab_size,
            layers,
            output_norm,
        })
    }

    /// Returns the recipe architecture tag bound to this cache.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Returns the hidden dimension bound to this cache.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Returns the vocabulary size bound to this cache.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns all per-layer pre-split repack descriptors.
    #[must_use]
    pub fn layers(&self) -> &[Phi3LayerRepack] {
        &self.layers
    }

    /// Returns the pre-split repack descriptor for `layer_idx`, if present.
    #[must_use]
    pub fn layer(&self, layer_idx: usize) -> Option<&Phi3LayerRepack> {
        self.layers.get(layer_idx)
    }

    /// Returns the decoded `f32` final `output_norm.weight` vector (`hidden_dim`).
    #[must_use]
    pub fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }
}

fn require_supported_tensor<'a>(
    recipe: &Phi3Recipe,
    weights: &'a dyn ModelWeights,
    name: &str,
) -> Result<TensorView<'a>, K3Error> {
    let view = weights.tensor(name)?;
    if !recipe.supports_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            recipe.architecture,
            view.name(),
            view.quant(),
        ));
    }
    Ok(view)
}

fn split_layer_qkv(
    recipe: &Phi3Recipe,
    layer: usize,
    weights: &dyn ModelWeights,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<(Phi3SplitMatrix, Phi3SplitMatrix, Phi3SplitMatrix), K3Error> {
    let qkv_name = format!("blk.{layer}.attn_qkv.weight");
    if let Ok(qkv) = weights.tensor(&qkv_name) {
        if !recipe.supports_quant(qkv.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                recipe.architecture,
                qkv.name(),
                qkv.quant(),
            ));
        }
        let expected_rows = q_dim + 2 * kv_dim;
        let (rows, _) = resolve_2d_dims(&qkv, hidden)?;
        if rows != expected_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: qkv.name().to_owned(),
                offset: qkv.offset(),
                byte_len: qkv.byte_len(),
                buffer_len: qkv.as_bytes().len(),
                reason: format!(
                    "fused attn_qkv row count {rows} != expected q_dim + 2*kv_dim ({expected_rows})"
                ),
            });
        }
        let wq = Phi3SplitMatrix::from_tensor_slice(
            &qkv,
            format!("blk.{layer}.attn_q.weight"),
            0,
            q_dim,
            hidden,
        )?;
        let wk = Phi3SplitMatrix::from_tensor_slice(
            &qkv,
            format!("blk.{layer}.attn_k.weight"),
            q_dim,
            kv_dim,
            hidden,
        )?;
        let wv = Phi3SplitMatrix::from_tensor_slice(
            &qkv,
            format!("blk.{layer}.attn_v.weight"),
            q_dim + kv_dim,
            kv_dim,
            hidden,
        )?;
        return Ok((wq, wk, wv));
    }

    let q_view = require_supported_tensor(recipe, weights, &format!("blk.{layer}.attn_q.weight"))?;
    let k_view = require_supported_tensor(recipe, weights, &format!("blk.{layer}.attn_k.weight"))?;
    let v_view = require_supported_tensor(recipe, weights, &format!("blk.{layer}.attn_v.weight"))?;
    Ok((
        Phi3SplitMatrix::from_tensor(&q_view, q_dim, hidden)?,
        Phi3SplitMatrix::from_tensor(&k_view, kv_dim, hidden)?,
        Phi3SplitMatrix::from_tensor(&v_view, kv_dim, hidden)?,
    ))
}

fn split_layer_mlp_up(
    recipe: &Phi3Recipe,
    layer: usize,
    weights: &dyn ModelWeights,
    hidden: usize,
    ffn_dim: usize,
) -> Result<(Phi3SplitMatrix, Phi3SplitMatrix), K3Error> {
    let ffn_up = require_supported_tensor(recipe, weights, &format!("blk.{layer}.ffn_up.weight"))?;
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    if let Ok(gate_view) = weights.tensor(&gate_name) {
        if !recipe.supports_quant(gate_view.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                recipe.architecture,
                gate_view.name(),
                gate_view.quant(),
            ));
        }
        return Ok((
            Phi3SplitMatrix::from_tensor(&gate_view, ffn_dim, hidden)?,
            Phi3SplitMatrix::from_tensor(&ffn_up, ffn_dim, hidden)?,
        ));
    }

    let expected_rows = ffn_dim * 2;
    let (rows, _) = resolve_2d_dims(&ffn_up, hidden)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: ffn_up.name().to_owned(),
            offset: ffn_up.offset(),
            byte_len: ffn_up.byte_len(),
            buffer_len: ffn_up.as_bytes().len(),
            reason: format!(
                "fused ffn_up row count {rows} != expected 2*ffn_dim ({expected_rows})"
            ),
        });
    }
    let wgate = Phi3SplitMatrix::from_tensor_slice(
        &ffn_up,
        format!("blk.{layer}.ffn_gate.weight"),
        0,
        ffn_dim,
        hidden,
    )?;
    let wup = Phi3SplitMatrix::from_tensor_slice(
        &ffn_up,
        format!("blk.{layer}.ffn_up.weight"),
        ffn_dim,
        ffn_dim,
        hidden,
    )?;
    Ok((wgate, wup))
}

#[cfg(feature = "with-kernels")]
const MAX_CACHED_LAYER_REPACKS: usize = 128;

#[cfg(feature = "with-kernels")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LayerRepackKey {
    architecture: &'static str,
    layer: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    digest: u64,
}

#[cfg(feature = "with-kernels")]
#[inline]
fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

#[cfg(feature = "with-kernels")]
fn hash_tensor_view(hash: &mut u64, view: &TensorView<'_>) {
    fnv1a_update(hash, view.name().as_bytes());
    fnv1a_update(hash, &[quant_tag(view.quant())]);
    for &dim in view.shape() {
        fnv1a_update(hash, &dim.to_le_bytes());
    }
    let raw = view.as_bytes();
    fnv1a_update(hash, &raw.len().to_le_bytes());
    if raw.len() <= 8192 {
        fnv1a_update(hash, raw);
    } else {
        fnv1a_update(hash, &(raw.as_ptr() as usize).to_le_bytes());
        let sample = 64_usize.min(raw.len());
        let step = (raw.len() / 16).max(sample);
        let mut off = 0_usize;
        while off + sample <= raw.len() {
            fnv1a_update(hash, &raw[off..off + sample]);
            off += step;
        }
        fnv1a_update(hash, &raw[raw.len() - sample..]);
    }
}

#[cfg(feature = "with-kernels")]
const fn quant_tag(q: QuantType) -> u8 {
    match q {
        QuantType::F32 => 0,
        QuantType::F16 => 1,
        QuantType::BF16 => 2,
        QuantType::Q4_0 => 3,
        QuantType::Q4_1 => 4,
        QuantType::Q5_0 => 5,
        QuantType::Q5_1 => 6,
        QuantType::Q8_0 => 7,
        QuantType::Q8_1 => 8,
        QuantType::Q2_K => 9,
        QuantType::Q3_K => 10,
        QuantType::Q4_K_M => 11,
        QuantType::Q5_K => 12,
        QuantType::Q6_K => 13,
        QuantType::IQ4_NL => 14,
        QuantType::MXFP4 => 15,
        QuantType::TQ1_0 => 16,
        QuantType::TQ2_0 => 17,
        QuantType::I2_S => 18,
    }
}

#[cfg(feature = "with-kernels")]
fn compute_layer_repack_key(
    recipe: &Phi3Recipe,
    layer: usize,
    weights: &dyn ModelWeights,
) -> LayerRepackKey {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    if let Ok(embd) = weights.tensor("token_embd.weight") {
        fnv1a_update(
            &mut hash,
            &(embd.as_bytes().as_ptr() as usize).to_le_bytes(),
        );
        fnv1a_update(&mut hash, &embd.as_bytes().len().to_le_bytes());
    }
    for suffix in [
        "attn_norm.weight",
        "ffn_norm.weight",
        "post_attention_norm.weight",
        "attn_qkv.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_gate.weight",
        "ffn_down.weight",
    ] {
        let name = format!("blk.{layer}.{suffix}");
        if let Ok(view) = weights.tensor(&name) {
            hash_tensor_view(&mut hash, &view);
        }
    }
    LayerRepackKey {
        architecture: recipe.architecture_id(),
        layer,
        hidden_dim: recipe.hidden_dim,
        q_dim: recipe.n_heads_q * recipe.head_dim,
        kv_dim: recipe.n_heads_kv * recipe.head_dim,
        ffn_dim: recipe.ffn_dim,
        digest: hash,
    }
}

#[cfg(feature = "with-kernels")]
fn global_layer_repack_cache() -> &'static RwLock<HashMap<LayerRepackKey, Arc<Phi3LayerRepack>>> {
    static CACHE: OnceLock<RwLock<HashMap<LayerRepackKey, Arc<Phi3LayerRepack>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Retrieves an ABA-safe cached [`Phi3LayerRepack`] for `layer` or pre-splits and caches it on
/// first prefill access.
#[cfg(feature = "with-kernels")]
pub(crate) fn get_or_repack_layer(
    recipe: &Phi3Recipe,
    layer: usize,
    weights: &dyn ModelWeights,
) -> Result<Arc<Phi3LayerRepack>, K3Error> {
    let key = compute_layer_repack_key(recipe, layer, weights);
    if let Ok(guard) = global_layer_repack_cache().read()
        && let Some(cached) = guard.get(&key)
    {
        return Ok(Arc::clone(cached));
    }
    let built = Arc::new(Phi3LayerRepack::from_weights(recipe, layer, weights)?);
    if let Ok(mut guard) = global_layer_repack_cache().write() {
        if guard.len() >= MAX_CACHED_LAYER_REPACKS {
            guard.clear();
        }
        guard.insert(key, Arc::clone(&built));
    }
    Ok(built)
}
