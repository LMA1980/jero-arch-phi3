//! Quantized block dequantization, geometry validation, and projection kernels for Phi-3 / Phi-4.
//!
//! Dispatches to `jero-kernels-rvv` hardware/scalar kernels when `with-kernels` is enabled, and
//! executes bit-equivalent portable scalar math across `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`,
//! `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` when built with `--no-default-features`.

use jero_core::{K3Error, QuantType, ScratchScope, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    AttentionMaskConfig, BaselineKernel, BlockQ8_0, BlockQ8K, Q4_0_BLOCK_BYTES,
    Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, Q8_K_BLOCK_ELEMENTS, VectorKernel, bf16_to_f32_finite,
    dequantize_token_row, f16_to_f32_finite, gemv_q4_0_q8_0_4x, gemv_q4_k_q8_k, gemv_q5_0_q8_0_4x,
    gemv_q5_k_q8_k, gemv_q6_k_q8_k, gemv_q8_0_q8_0_4x, gqa_causal_attention_prefill_f32,
    gqa_causal_attention_with_scratch_f32, quantize_row_q8_0_f32, quantize_row_q8_0_in_scope,
    quantize_row_q8_k, quantize_row_q8_k_in_scope, swiglu_f32, vec_dot_bf16, vec_dot_f16,
    vec_dot_q4_0, vec_dot_q5_0,
};
#[cfg(all(feature = "with-kernels", not(miri)))]
use jero_kernels_rvv::{gqa_causal_attention_prefill_kv_heads_f32, gqa_kv_head_partition};

#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_BYTES: usize = 18;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_BYTES: usize = 34;
#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_BYTES: usize = 144;
#[cfg(not(feature = "with-kernels"))]
const Q5_K_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q5_K_BLOCK_BYTES: usize = 176;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of scalar elements per `Q5_0` block.
const Q5_0_BLOCK_ELEMENTS: usize = 32;
/// Storage size in bytes of a single `Q5_0` block (`2B f16` scale + `4B qh` + `16B qs`).
const Q5_0_BLOCK_BYTES: usize = 22;

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn f16_to_f32_finite(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    let e = exp + 127 - 15;
    f32::from_bits((sign << 31) | (e << 23) | (mant << 13))
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

/// Sanitizes non-finite or extreme synthetic byte patterns decoded as raw `f32`.
#[inline]
pub(crate) fn sanitize_f32_weight(val: f32) -> f32 {
    if !val.is_finite() {
        0.0
    } else if val.abs() > 1.0e6 {
        val.signum() * 0.0625
    } else {
        val
    }
}

/// Returns `(block_elements, block_bytes)` for all Phi-3 / Phi-4 admitted formats.
#[must_use]
pub(crate) const fn quant_block_geometry(quant: QuantType) -> Option<(usize, usize)> {
    match quant {
        QuantType::F32 => Some((1, 4)),
        QuantType::F16 | QuantType::BF16 => Some((1, 2)),
        QuantType::Q4_0 => Some((Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES)),
        QuantType::Q5_0 => Some((Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES)),
        QuantType::Q8_0 => Some((Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES)),
        QuantType::Q4_K_M => Some((Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES)),
        QuantType::Q5_K => Some((Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES)),
        QuantType::Q6_K => Some((Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES)),
        _ => None,
    }
}

/// Computes the byte stride of one contiguous row of `cols` elements for `quant`.
#[must_use]
pub(crate) const fn quant_row_bytes(cols: usize, quant: QuantType) -> Option<usize> {
    let Some((blk_elems, blk_bytes)) = quant_block_geometry(quant) else {
        return None;
    };
    if cols == 0 || cols % blk_elems != 0 {
        return None;
    }
    (cols / blk_elems).checked_mul(blk_bytes)
}

/// Resolves `(rows, cols)` of a 2D tensor given `expected_cols` (`in_features`),
/// accepting both GGUF `[cols, rows]` (`ne[0] = cols`) and `[rows, cols]`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if `tensor`
/// is non-2D, does not match `expected_cols`, uses an unsupported quant, or lacks bytes.
pub(crate) fn resolve_2d_dims(
    tensor: &TensorView<'_>,
    expected_cols: usize,
) -> Result<(usize, usize), K3Error> {
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
    let (d0, d1) = (shape[0], shape[1]);
    let (rows, cols) = if d0 == expected_cols {
        (d1, d0)
    } else if d1 == expected_cols {
        (d0, d1)
    } else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "2D tensor shape {shape:?} does not match expected cols {expected_cols}"
            ),
        });
    };

    let row_bytes = quant_row_bytes(cols, tensor.quant())
        .ok_or_else(|| K3Error::unsupported_tensor_quant("phi3", tensor.name(), tensor.quant()))?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: "matrix byte length overflow".to_owned(),
        })?;
    if tensor.as_bytes().len() < expected_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: expected_bytes,
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "tensor byte length {} smaller than required {expected_bytes}",
                tensor.as_bytes().len()
            ),
        });
    }
    Ok((rows, cols))
}

/// Dequantizes a 22-byte `Q5_0` block into 32 `f32` elements matching GGML `dequantize_row_q5_0`.
pub(crate) fn dequantize_q5_0_block(
    block: &[u8; Q5_0_BLOCK_BYTES],
    out: &mut [f32; Q5_0_BLOCK_ELEMENTS],
) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for (j, &packed) in qs.iter().enumerate() {
        let xh0 = u8::try_from(((qh >> j) & 1) << 4).unwrap_or(0);
        let xh1 = u8::try_from(((qh >> (j + 16)) & 1) << 4).unwrap_or(0);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "5-bit unsigned 0..31 - 16 fits i8"
        )]
        let q0 = ((packed & 0x0F) | xh0) as i8 - 16;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "5-bit unsigned 0..31 - 16 fits i8"
        )]
        let q1 = (((packed >> 4) & 0x0F) | xh1) as i8 - 16;
        out[j] = d * f32::from(q0);
        out[j + 16] = d * f32::from(q1);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q4_0_block(block: &[u8; Q4_0_BLOCK_BYTES], out: &mut [f32; Q4_0_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for (i, &byte) in block[2..18].iter().enumerate() {
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble 0..15 - 8 fits i8")]
        let q0 = (byte & 0x0F) as i8 - 8;
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble 0..15 - 8 fits i8")]
        let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
        out[i] = d * f32::from(q0);
        out[i + 16] = d * f32::from(q1);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q8_0_block(block: &[u8; Q8_0_BLOCK_BYTES], out: &mut [f32; Q8_0_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for (dst, &byte) in out.iter_mut().zip(&block[2..34]) {
        #[expect(clippy::cast_possible_wrap, reason = "Q8_0 byte represents signed i8")]
        let q = byte as i8;
        *dst = d * f32::from(q);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn unpack_k_scales_6bit(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0_u8; 8];
    let mut m = [0_u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q4_k_m_block(
    block: &[u8; Q4_K_M_BLOCK_BYTES],
    out: &mut [f32; Q4_K_M_BLOCK_ELEMENTS],
) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }
    let (sc, m) = unpack_k_scales_6bit(&block[4..16]);
    let qs = &block[16..144];
    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);
        let out_idx = chunk * 64;
        let q_offset = chunk * 32;
        for l in 0..32 {
            out[out_idx + l] = d1 * f32::from(qs[q_offset + l] & 0x0F) - m1;
            out[out_idx + 32 + l] = d2 * f32::from((qs[q_offset + l] >> 4) & 0x0F) - m2;
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q5_k_block(block: &[u8; Q5_K_BLOCK_BYTES], out: &mut [f32; Q5_K_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }
    let (sc, m) = unpack_k_scales_6bit(&block[4..16]);
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut u1 = 1_u8;
    let mut u2 = 2_u8;
    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);
        let out_idx = chunk * 64;
        let q_offset = chunk * 32;
        for l in 0..32 {
            let hb1 = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
            let hb2 = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
            out[out_idx + l] = d1 * f32::from((qs[q_offset + l] & 0x0F) + hb1) - m1;
            out[out_idx + 32 + l] = d2 * f32::from(((qs[q_offset + l] >> 4) & 0x0F) + hb2) - m2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q6_k_block(block: &[u8; Q6_K_BLOCK_BYTES], out: &mut [f32; Q6_K_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[208], block[209]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    for (half_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
        let low_base = half_idx * 64;
        let high_base = half_idx * 32;
        let sc_base = half_idx * 8;
        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q1 = (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q2 = (((ql[low_base + l + 32] & 0x0F) | (((qh[high_base + l] >> 2) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q3 = ((((ql[low_base + l] >> 4) & 0x0F) | (((qh[high_base + l] >> 4) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q4 = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                | (((qh[high_base + l] >> 6) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc1 = f32::from(scales[sc_base + is] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc2 = f32::from(scales[sc_base + is + 2] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc3 = f32::from(scales[sc_base + is + 4] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc4 = f32::from(scales[sc_base + is + 6] as i8);
            out[n + l] = d * sc1 * f32::from(q1);
            out[n + l + 32] = d * sc2 * f32::from(q2);
            out[n + l + 64] = d * sc3 * f32::from(q3);
            out[n + l + 96] = d * sc4 * f32::from(q4);
        }
    }
}

#[inline]
pub(crate) fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}

fn dequant_block_row_payload<const ELEMS: usize, const BYTES: usize>(
    tensor_name: &str,
    row_raw: &[u8],
    out: &mut [f32],
    decode: fn(&[u8; BYTES], &mut [f32; ELEMS]),
) -> Result<(), K3Error> {
    let mut block_out = [0.0_f32; ELEMS];
    for (blk_idx, out_chunk) in out.chunks_exact_mut(ELEMS).enumerate() {
        let start = blk_idx * BYTES;
        let block: &[u8; BYTES] =
            row_raw[start..start + BYTES]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: tensor_name.to_owned(),
                    offset: start,
                    byte_len: BYTES,
                    buffer_len: row_raw.len(),
                    reason: "quantized block slice length mismatch".to_owned(),
                })?;
        decode(block, &mut block_out);
        out_chunk.copy_from_slice(&block_out);
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn dot_block_rows<const ELEMS: usize, const BYTES: usize>(
    tensor: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    decode: fn(&[u8; BYTES], &mut [f32; ELEMS]),
) -> Result<(), K3Error> {
    let blocks_per_row = cols / ELEMS;
    let row_bytes = blocks_per_row * BYTES;
    let raw = tensor.as_bytes();
    let mut block_out = [0.0_f32; ELEMS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_raw = &raw[row * row_bytes..(row + 1) * row_bytes];
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let start = blk_idx * BYTES;
            let block: &[u8; BYTES] = row_raw[start..start + BYTES].try_into().map_err(|_| {
                K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: row * row_bytes + start,
                    byte_len: BYTES,
                    buffer_len: raw.len(),
                    reason: "quantized block slice length mismatch".to_owned(),
                }
            })?;
            decode(block, &mut block_out);
            let in_chunk = &input[blk_idx * ELEMS..(blk_idx + 1) * ELEMS];
            row_sum += dot_product_f32(&block_out, in_chunk);
        }
        *out_slot = row_sum;
    }
    Ok(())
}

/// Decodes a single 1D normalization tensor (`F32`, `F16`, or `BF16`) into `out` (`expected_len`
/// elements) without heap allocation.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if element count, output slice length, or byte length
/// mismatches, or [`K3Error::CapabilityMismatch`] if `tensor.quant()` is not an unquantized
/// norm format.
pub(crate) fn decode_1d_norm_into(
    tensor: &TensorView<'_>,
    expected_len: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if tensor.num_elements() != expected_len || out.len() != expected_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "norm element count {} (out len {}) != expected hidden_dim {expected_len}",
                tensor.num_elements(),
                out.len()
            ),
        });
    }
    decode_unquant_payload(tensor, tensor.as_bytes(), out)
}

/// Decodes a single 1D normalization tensor (`F32`, `F16`, or `BF16`) into `expected_len` floats.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if the element count or byte length mismatches, or
/// [`K3Error::CapabilityMismatch`] if `tensor.quant()` is not an unquantized norm format.
pub(crate) fn decode_1d_norm(
    tensor: &TensorView<'_>,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; expected_len];
    decode_1d_norm_into(tensor, expected_len, &mut out)?;
    Ok(out)
}

fn decode_unquant_payload(
    tensor: &TensorView<'_>,
    raw: &[u8],
    out: &mut [f32],
) -> Result<(), K3Error> {
    let elem_bytes = match tensor.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "phi3".to_owned(),
                reason: format!(
                    "tensor '{}' uses unsupported unquantized format {:?}",
                    tensor.name(),
                    other
                ),
            });
        }
    };
    let required = out.len().saturating_mul(elem_bytes);
    if raw.len() < required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: required,
            buffer_len: raw.len(),
            reason: format!(
                "buffer len {} smaller than required {required} bytes",
                raw.len()
            ),
        });
    }
    match tensor.quant() {
        QuantType::F32 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(4)) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                *slot = sanitize_f32_weight(val);
            }
        }
        QuantType::F16 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(2)) {
                *slot = f16_to_f32_finite(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        QuantType::BF16 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(2)) {
                *slot = bf16_to_f32_finite(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Decodes a single row (`cols` elements) from `tensor` (`token_embd.weight` or weight matrix)
/// across all admitted formats (`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, `Q6_K`).
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on out-of-range row/shape mismatch or
/// [`K3Error::CapabilityMismatch`] on unsupported quantization types.
pub(crate) fn dequantize_row_to_f32(
    tensor: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if out.len() != cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("output row len {} != cols {cols}", out.len()),
        });
    }

    let (num_rows, _) = resolve_2d_dims(tensor, cols)?;
    if row >= num_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: row,
            byte_len: cols,
            buffer_len: num_rows,
            reason: format!("row index {row} out of bounds for {num_rows} rows"),
        });
    }

    #[cfg(feature = "with-kernels")]
    if tensor.quant() != QuantType::Q5_0 {
        return dequantize_token_row(tensor, row, out);
    }

    let row_bytes = quant_row_bytes(cols, tensor.quant())
        .ok_or_else(|| K3Error::unsupported_tensor_quant("phi3", tensor.name(), tensor.quant()))?;
    let start = row * row_bytes;
    let row_raw = &tensor.as_bytes()[start..start + row_bytes];

    match tensor.quant() {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            decode_unquant_payload(tensor, row_raw, out)
        }
        QuantType::Q5_0 => dequant_block_row_payload::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q5_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q4_0 => dequant_block_row_payload::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q4_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q8_0 => dequant_block_row_payload::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q8_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q4_K_M => {
            dequant_block_row_payload::<Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES>(
                tensor.name(),
                row_raw,
                out,
                dequantize_q4_k_m_block,
            )
        }
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q5_K => dequant_block_row_payload::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q5_k_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q6_K => dequant_block_row_payload::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q6_k_block,
        ),
        #[cfg(feature = "with-kernels")]
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            tensor.name(),
            other,
        )),
        #[cfg(not(feature = "with-kernels"))]
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            tensor.name(),
            other,
        )),
    }
}

#[cfg(feature = "with-kernels")]
const MAX_STACK_Q8_ELEMENTS: usize = 8192;
#[cfg(feature = "with-kernels")]
const MAX_STACK_Q8_0_BLOCKS: usize = MAX_STACK_Q8_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
#[cfg(feature = "with-kernels")]
const MAX_STACK_Q8_K_BLOCKS: usize = MAX_STACK_Q8_ELEMENTS / Q8_K_BLOCK_ELEMENTS;

/// Reusable stack-first `Q8_0` / `Q8_K` quantized activation buffers for `M = 1` projections
/// sharing the same input vector (such as split `(Wq, Wk, Wv)` and `(Wgate, Wup)`).
#[cfg(feature = "with-kernels")]
pub(crate) struct SharedQ8Input<'a> {
    input: &'a [f32],
    q8_0_stack: [BlockQ8_0; MAX_STACK_Q8_0_BLOCKS],
    q8_0_heap: Vec<BlockQ8_0>,
    q8_0_len: usize,
    q8_k_stack: [BlockQ8K; MAX_STACK_Q8_K_BLOCKS],
    q8_k_heap: Vec<BlockQ8K>,
    q8_k_len: usize,
}

#[cfg(feature = "with-kernels")]
impl<'a> SharedQ8Input<'a> {
    pub(crate) fn new(input: &'a [f32]) -> Self {
        Self {
            input,
            q8_0_stack: [BlockQ8_0::ZERO; MAX_STACK_Q8_0_BLOCKS],
            q8_0_heap: Vec::new(),
            q8_0_len: 0,
            q8_k_stack: [BlockQ8K::ZERO; MAX_STACK_Q8_K_BLOCKS],
            q8_k_heap: Vec::new(),
            q8_k_len: 0,
        }
    }

    fn ensure_q8_0(&mut self) -> Result<(), K3Error> {
        if self.q8_0_len > 0 {
            return Ok(());
        }
        let blocks = self.input.len() / Q8_0_BLOCK_ELEMENTS;
        if blocks <= MAX_STACK_Q8_0_BLOCKS {
            quantize_row_q8_0_f32(self.input, &mut self.q8_0_stack[..blocks])?;
        } else {
            self.q8_0_heap.resize(blocks, BlockQ8_0::ZERO);
            quantize_row_q8_0_f32(self.input, &mut self.q8_0_heap)?;
        }
        self.q8_0_len = blocks;
        Ok(())
    }

    fn ensure_q8_k(&mut self) -> Result<(), K3Error> {
        if self.q8_k_len > 0 {
            return Ok(());
        }
        let blocks = self.input.len() / Q8_K_BLOCK_ELEMENTS;
        if blocks <= MAX_STACK_Q8_K_BLOCKS {
            quantize_row_q8_k(self.input, &mut self.q8_k_stack[..blocks])?;
        } else {
            self.q8_k_heap.resize(blocks, BlockQ8K::ZERO);
            quantize_row_q8_k(self.input, &mut self.q8_k_heap)?;
        }
        self.q8_k_len = blocks;
        Ok(())
    }

    fn q8_0_slice(&self) -> &[BlockQ8_0] {
        if self.q8_0_heap.is_empty() {
            &self.q8_0_stack[..self.q8_0_len]
        } else {
            &self.q8_0_heap
        }
    }

    fn q8_k_slice(&self) -> &[BlockQ8K] {
        if self.q8_k_heap.is_empty() {
            &self.q8_k_stack[..self.q8_k_len]
        } else {
            &self.q8_k_heap
        }
    }
}

/// Evaluates `f` over a zero-allocation `[row_start..row_start + row_count)` 2D view normalized
/// to `[cols, row_count]` without heap allocations.
pub(crate) fn with_2d_rows_view<R>(
    tensor: &TensorView<'_>,
    row_start: usize,
    row_count: usize,
    expected_cols: usize,
    f: impl FnOnce(&TensorView<'_>) -> Result<R, K3Error>,
) -> Result<R, K3Error> {
    let (total_rows, cols) = resolve_2d_dims(tensor, expected_cols)?;
    let row_end = row_start.saturating_add(row_count);
    if row_count == 0 || row_end > total_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: row_start,
            byte_len: row_count,
            buffer_len: total_rows,
            reason: format!(
                "row slice [{row_start}..{row_end}) out of bounds for {total_rows} rows"
            ),
        });
    }
    let row_bytes = quant_row_bytes(cols, tensor.quant())
        .ok_or_else(|| K3Error::unsupported_tensor_quant("phi3", tensor.name(), tensor.quant()))?;
    let byte_start = row_start * row_bytes;
    let byte_end = row_end * row_bytes;
    let shape = [cols, row_count];
    let view = TensorView::from_bytes(
        tensor.name(),
        tensor.quant(),
        &shape,
        0,
        &tensor.as_bytes()[byte_start..byte_end],
    )?;
    f(&view)
}

/// Dispatches row-partitioned `M = 1` work across active Rayon worker threads with zero heap
/// allocations and deterministic bit-identical outputs across any worker count (`not(miri)`).
pub(crate) fn parallel_split_rows<F>(
    out: &mut [f32],
    align: usize,
    min_parallel_rows: usize,
    op: &F,
) -> Result<(), K3Error>
where
    F: Fn(&mut [f32], usize) -> Result<(), K3Error> + Sync,
{
    #[cfg(miri)]
    {
        let _ = (align, min_parallel_rows);
        op(out, 0)
    }
    #[cfg(not(miri))]
    {
        let step = align.max(1);
        let workers = rayon::current_num_threads();
        if workers <= 1 || out.len() < min_parallel_rows.max(step.saturating_mul(2)) {
            return op(out, 0);
        }
        parallel_split_rows_inner(out, 0, step, workers, op)
    }
}

#[cfg(not(miri))]
fn parallel_split_rows_inner<F>(
    out: &mut [f32],
    base_row: usize,
    align: usize,
    budget_workers: usize,
    op: &F,
) -> Result<(), K3Error>
where
    F: Fn(&mut [f32], usize) -> Result<(), K3Error> + Sync,
{
    let total_units = out.len().div_ceil(align);
    if budget_workers <= 1 || total_units <= 1 {
        return op(out, base_row);
    }
    let left_workers = budget_workers / 2;
    let right_workers = budget_workers - left_workers;
    let left_units = (total_units * left_workers / budget_workers)
        .max(1)
        .min(total_units - 1);
    let mid_rows = left_units * align;
    let (left_out, right_out) = out.split_at_mut(mid_rows);
    let (left_res, right_res) = rayon::join(
        || parallel_split_rows_inner(left_out, base_row, align, left_workers, op),
        || parallel_split_rows_inner(right_out, base_row + mid_rows, align, right_workers, op),
    );
    left_res?;
    right_res
}

/// Multi-row `M = 1` parallel LM-head projection dispatching to `gemv_q*_q8_*` multi-row integer
/// RVV/scalar kernels when `with-kernels` is enabled, or row-parallel scalar math under
/// `--no-default-features` (`#34`).
pub(crate) fn project_lm_head_parallel(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    scope: Option<&ScratchScope<'_>>,
) -> Result<(), K3Error> {
    let expected_cols = input.len();
    let expected_rows = output.len();
    let (rows, cols) = resolve_2d_dims(weights, expected_cols)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "weight matrix output rows {rows} != expected output len {expected_rows}"
            ),
        });
    }

    #[cfg(feature = "with-kernels")]
    {
        project_lm_head_q8_parallel(output, weights, cols, input, scope)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = scope;
        parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
            with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                project_matvec_scalar(sub_out, sub_view, sub_out.len(), cols, input)
            })
        })
    }
}

/// Validates shape and computes `output[0..expected_rows] = weights * input[0..expected_cols]`
/// across all admitted Phi-3 / Phi-4 quantization types (`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`,
/// `Q8_0`, `Q4_K_M`, `Q5_K`, `Q6_K`), partitioning output rows across active `WorkerPool` threads.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on shape/buffer mismatch or
/// [`K3Error::CapabilityMismatch`] on unsupported quantization formats.
pub(crate) fn project_matvec(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let expected_cols = input.len();
    let expected_rows = output.len();
    let (rows, cols) = resolve_2d_dims(weights, expected_cols)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "weight matrix output rows {rows} != expected output len {expected_rows}"
            ),
        });
    }

    #[cfg(feature = "with-kernels")]
    {
        parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
            with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                project_matvec_kernels(sub_out, sub_view, cols, input)
            })
        })
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
            with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                project_matvec_scalar(sub_out, sub_view, sub_out.len(), cols, input)
            })
        })
    }
}

#[cfg(feature = "with-kernels")]
fn project_matvec_kernels(
    output: &mut [f32],
    weights: &TensorView<'_>,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(output, weights, input),
        QuantType::Q8_0 => BaselineKernel::new().vec_dot_q8_0(output, weights, input),
        QuantType::Q6_K => BaselineKernel::new().vec_dot_q6_k(output, weights, input),
        QuantType::Q5_K => BaselineKernel::new().vec_dot_q5_k(output, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(output, weights, input),
        QuantType::Q5_0 => vec_dot_q5_0(output, weights, input),
        QuantType::F16 => vec_dot_f16(output, weights, input),
        QuantType::BF16 => vec_dot_bf16(output, weights, input),
        QuantType::F32 => dot_unquantized_rows(output, weights, output.len(), cols, input),
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            weights.name(),
            other,
        )),
    }
}

/// Projects `[row_start..row_start + out.len()]` of `weights` against a shared pre-quantized `Q8`
/// input activation without re-quantizing across split `(Wq, Wk, Wv)` or `(Wgate, Wup)` tensors.
#[cfg(feature = "with-kernels")]
pub(crate) fn project_matvec_shared_q8(
    output: &mut [f32],
    weights: &TensorView<'_>,
    shared: &mut SharedQ8Input<'_>,
) -> Result<(), K3Error> {
    let expected_cols = shared.input.len();
    let (rows, cols) = resolve_2d_dims(weights, expected_cols)?;
    if rows != output.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "weight matrix output rows {rows} != expected output len {}",
                output.len()
            ),
        });
    }
    match weights.quant() {
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            shared.ensure_q8_k()?;
            let q8_k = shared.q8_k_slice();
            parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                    dispatch_q8_k_gemv(sub_out, sub_view, q8_k)
                })
            })
        }
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            shared.ensure_q8_0()?;
            let q8_0 = shared.q8_0_slice();
            parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                    dispatch_q8_0_gemv(sub_out, sub_view, q8_0)
                })
            })
        }
        _ => {
            let input = shared.input;
            parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                    dispatch_unquant_or_f32_gemv(sub_out, sub_view, cols, input)
                })
            })
        }
    }
}

#[cfg(feature = "with-kernels")]
fn project_lm_head_q8_parallel(
    output: &mut [f32],
    weights: &TensorView<'_>,
    cols: usize,
    input: &[f32],
    scope: Option<&ScratchScope<'_>>,
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            if let Some(scratch_scope) = scope {
                let q8_k = quantize_row_q8_k_in_scope(input, scratch_scope)?;
                return parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                    with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                        dispatch_q8_k_gemv(sub_out, sub_view, q8_k)
                    })
                });
            }
            let blocks = cols / Q8_K_BLOCK_ELEMENTS;
            if blocks <= MAX_STACK_Q8_K_BLOCKS {
                let mut stack = [BlockQ8K::ZERO; MAX_STACK_Q8_K_BLOCKS];
                let q8_k = &mut stack[..blocks];
                quantize_row_q8_k(input, q8_k)?;
                return parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                    with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                        dispatch_q8_k_gemv(sub_out, sub_view, q8_k)
                    })
                });
            }
            let mut heap = vec![BlockQ8K::ZERO; blocks];
            quantize_row_q8_k(input, &mut heap)?;
            parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                    dispatch_q8_k_gemv(sub_out, sub_view, &heap)
                })
            })
        }
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            if let Some(scratch_scope) = scope {
                let q8_0 = quantize_row_q8_0_in_scope(input, scratch_scope)?;
                return parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                    with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                        dispatch_q8_0_gemv(sub_out, sub_view, q8_0)
                    })
                });
            }
            let blocks = cols / Q8_0_BLOCK_ELEMENTS;
            if blocks <= MAX_STACK_Q8_0_BLOCKS {
                let mut stack = [BlockQ8_0::ZERO; MAX_STACK_Q8_0_BLOCKS];
                let q8_0 = &mut stack[..blocks];
                quantize_row_q8_0_f32(input, q8_0)?;
                return parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                    with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                        dispatch_q8_0_gemv(sub_out, sub_view, q8_0)
                    })
                });
            }
            let mut heap = vec![BlockQ8_0::ZERO; blocks];
            quantize_row_q8_0_f32(input, &mut heap)?;
            parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
                with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                    dispatch_q8_0_gemv(sub_out, sub_view, &heap)
                })
            })
        }
        _ => parallel_split_rows(output, 4, 8, &|sub_out, row_start| {
            with_2d_rows_view(weights, row_start, sub_out.len(), cols, |sub_view| {
                dispatch_unquant_or_f32_gemv(sub_out, sub_view, cols, input)
            })
        }),
    }
}

#[cfg(feature = "with-kernels")]
fn dispatch_q8_k_gemv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    q8_k: &[BlockQ8K],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_K_M => gemv_q4_k_q8_k(output, weights, q8_k),
        QuantType::Q5_K => gemv_q5_k_q8_k(output, weights, q8_k),
        QuantType::Q6_K => gemv_q6_k_q8_k(output, weights, q8_k),
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            weights.name(),
            other,
        )),
    }
}

#[cfg(feature = "with-kernels")]
fn dispatch_q8_0_gemv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    q8_0: &[BlockQ8_0],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_0 => gemv_q4_0_q8_0_4x(output, weights, q8_0),
        QuantType::Q5_0 => gemv_q5_0_q8_0_4x(output, weights, q8_0),
        QuantType::Q8_0 => gemv_q8_0_q8_0_4x(output, weights, q8_0),
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            weights.name(),
            other,
        )),
    }
}

#[cfg(feature = "with-kernels")]
fn dispatch_unquant_or_f32_gemv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::F16 => vec_dot_f16(output, weights, input),
        QuantType::BF16 => vec_dot_bf16(output, weights, input),
        QuantType::F32 => dot_unquantized_rows(output, weights, output.len(), cols, input),
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            weights.name(),
            other,
        )),
    }
}

#[cfg(not(feature = "with-kernels"))]
fn project_matvec_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            dot_unquantized_rows(output, weights, rows, cols, input)
        }
        QuantType::Q4_0 => dot_block_rows::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q4_0_block,
        ),
        QuantType::Q5_0 => dot_block_rows::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_0_block,
        ),
        QuantType::Q8_0 => dot_block_rows::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q8_0_block,
        ),
        QuantType::Q4_K_M => dot_block_rows::<Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q4_k_m_block,
        ),
        QuantType::Q5_K => dot_block_rows::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_k_block,
        ),
        QuantType::Q6_K => dot_block_rows::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q6_k_block,
        ),
        other => Err(K3Error::unsupported_tensor_quant(
            "phi3",
            weights.name(),
            other,
        )),
    }
}

fn dot_unquantized_rows(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    let elem_bytes = match weights.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::unsupported_tensor_quant(
                "phi3",
                weights.name(),
                other,
            ));
        }
    };
    let stride = cols * elem_bytes;
    let raw = weights.as_bytes();
    let required = rows.saturating_mul(stride);
    if raw.len() < required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: required,
            buffer_len: raw.len(),
            reason: format!(
                "buffer len {} smaller than required {required} bytes",
                raw.len()
            ),
        });
    }
    let mut chunk_buf = [0.0_f32; 256];
    for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_raw = &raw[r * stride..(r + 1) * stride];
        let mut acc = 0.0_f32;
        for (chunk_idx, in_chunk) in input[..cols].chunks(chunk_buf.len()).enumerate() {
            let chunk_elems = in_chunk.len();
            let byte_start = chunk_idx * chunk_buf.len() * elem_bytes;
            let byte_end = byte_start + chunk_elems * elem_bytes;
            decode_unquant_payload(
                weights,
                &row_raw[byte_start..byte_end],
                &mut chunk_buf[..chunk_elems],
            )?;
            #[cfg(feature = "with-kernels")]
            {
                acc += dot_product_f32(&chunk_buf[..chunk_elems], in_chunk);
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                for (&w, &x) in chunk_buf[..chunk_elems].iter().zip(in_chunk) {
                    acc += w * x;
                }
            }
        }
        *out_slot = acc;
    }
    Ok(())
}

/// Computes single-vector `RMSNorm` (`output[i] = input[i] / sqrt(mean(input^2) + eps) * weight[i]`).
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch or `input` is empty.
pub(crate) fn rmsnorm_f32(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(output, input, weight, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if input.is_empty() || output.len() != input.len() || weight.len() != input.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rmsnorm".to_owned(),
                offset: 0,
                byte_len: size_of_val(output),
                buffer_len: size_of_val(input),
                reason: format!(
                    "rmsnorm length mismatch (out={}, in={}, w={})",
                    output.len(),
                    input.len(),
                    weight.len()
                ),
            });
        }
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        #[expect(clippy::cast_precision_loss, reason = "hidden_dim fits cleanly in f32")]
        let inv_rms = 1.0_f32 / (sum_sq / (input.len() as f32) + eps).sqrt();
        for ((dst, &x), &w) in output.iter_mut().zip(input).zip(weight) {
            *dst = x * inv_rms * w;
        }
        Ok(())
    }
}

/// Computes `SwiGLU` activation (`out[i] = silu(gate[i]) * up[i]`).
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on slice length mismatch.
pub(crate) fn swiglu_layer_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        swiglu_f32(out, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if out.len() != gate.len() || gate.len() != up.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "swiglu".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(gate),
                reason: format!(
                    "swiglu length mismatch (out={}, gate={}, up={})",
                    out.len(),
                    gate.len(),
                    up.len()
                ),
            });
        }
        for ((dst, &g), &u) in out.iter_mut().zip(gate).zip(up) {
            let sig = 1.0_f32 / (1.0_f32 + (-g).exp());
            *dst = (g * sig) * u;
        }
        Ok(())
    }
}

/// Computes zero-allocation causal Grouped-Query Attention across `[0..seq_len)` cached KV slots
/// using a caller-supplied `scores_scratch` slice (`len >= seq_len`).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on invalid
/// geometry or buffer dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_with_scratch_f32 kernel ABI directly"
)]
pub(crate) fn gqa_attention_with_scratch_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_with_scratch_f32(
            out,
            q,
            k_cache,
            v_cache,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            scale,
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || seq_len == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "phi3".to_owned(),
                reason: "GQA dimensions must be non-zero".to_owned(),
            });
        }
        if n_heads_q % n_heads_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "phi3".to_owned(),
                reason: format!(
                    "n_heads_q ({n_heads_q}) not divisible by n_heads_kv ({n_heads_kv})"
                ),
            });
        }
        let q_len = n_heads_q * head_dim;
        let kv_min = seq_len * n_heads_kv * head_dim;
        if q.len() != q_len
            || out.len() != q_len
            || k_cache.len() < kv_min
            || v_cache.len() < kv_min
            || scores_scratch.len() < seq_len
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "gqa_causal_attention".to_owned(),
                offset: 0,
                byte_len: size_of_val(q),
                buffer_len: size_of_val(scores_scratch),
                reason: "GQA query, KV cache, or scores_scratch buffer dimension mismatch"
                    .to_owned(),
            });
        }
        let group_size = n_heads_q / n_heads_kv;
        let scores = &mut scores_scratch[..seq_len];
        for h_q in 0..n_heads_q {
            let h_kv = h_q / group_size;
            let q_head = &q[h_q * head_dim..(h_q + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for (s, slot) in scores.iter_mut().enumerate().take(seq_len) {
                let off = (s * n_heads_kv + h_kv) * head_dim;
                let sc = dot_product_f32(q_head, &k_cache[off..off + head_dim]) * scale;
                *slot = sc;
                if sc > max_score {
                    max_score = sc;
                }
            }
            let mut sum_exp = 0.0_f32;
            for sc in &mut *scores {
                let e = (*sc - max_score).exp();
                *sc = e;
                sum_exp += e;
            }
            #[expect(clippy::cast_precision_loss, reason = "seq_len fits cleanly in f32")]
            let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
                1.0_f32 / sum_exp
            } else {
                1.0_f32 / (seq_len as f32)
            };
            let out_head = &mut out[h_q * head_dim..(h_q + 1) * head_dim];
            out_head.fill(0.0);
            for (s, &e) in scores.iter().enumerate().take(seq_len) {
                let w = e * inv_sum;
                let off = (s * n_heads_kv + h_kv) * head_dim;
                let v_head = &v_cache[off..off + head_dim];
                for (dst, &v_val) in out_head.iter_mut().zip(v_head) {
                    *dst += w * v_val;
                }
            }
        }
        Ok(())
    }
}

/// Computes batched `RMSNorm` across `m` contiguous rows of width `hidden`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if dimensions are zero or slice lengths mismatch.
pub(crate) fn rmsnorm_batch_f32(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    m: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), K3Error> {
    let expected = m.checked_mul(hidden);
    if m == 0
        || hidden == 0
        || expected != Some(input.len())
        || expected != Some(output.len())
        || weight.len() != hidden
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rmsnorm_batch".to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!(
                "rmsnorm_batch mismatch (m={m}, hidden={hidden}, out={}, in={}, w={})",
                output.len(),
                input.len(),
                weight.len()
            ),
        });
    }

    #[cfg(not(miri))]
    if m >= 2 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        return output
            .par_chunks_exact_mut(hidden)
            .enumerate()
            .try_for_each(|(row_idx, out_row)| {
                let in_row = &input[row_idx * hidden..(row_idx + 1) * hidden];
                rmsnorm_f32(out_row, in_row, weight, eps)
            });
    }

    for m_idx in 0..m {
        let start = m_idx * hidden;
        let end = start + hidden;
        rmsnorm_f32(&mut output[start..end], &input[start..end], weight, eps)?;
    }
    Ok(())
}

#[cfg(all(feature = "with-kernels", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_kv_heads_f32 ABI"
)]
fn gqa_prefill_causal_parallel(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    sliding_window: Option<usize>,
    config: AttentionMaskConfig,
    workers: usize,
) -> Result<(), K3Error> {
    use rayon::prelude::*;
    let scratch_len = sliding_window.map_or(total_seq_len, |w| w.min(total_seq_len));
    let full_q_width = n_heads_q * head_dim;
    let partials = (0..workers)
        .into_par_iter()
        .filter_map(|worker_idx| gqa_kv_head_partition(n_heads_kv, n_heads_q, worker_idx, workers))
        .map(|(kv_range, q_range)| {
            let sub_q_width = q_range.len() * head_dim;
            let mut sub_out = vec![0.0_f32; m * sub_q_width];
            let mut local_scratch = vec![0.0_f32; scratch_len.max(1)];
            gqa_causal_attention_prefill_kv_heads_f32(
                &mut sub_out,
                q,
                k_cache,
                v_cache,
                m,
                n_heads_q,
                n_heads_kv,
                head_dim,
                total_seq_len,
                kv_range,
                config,
                &mut local_scratch,
            )?;
            Ok((q_range, sub_out))
        })
        .collect::<Result<Vec<_>, K3Error>>()?;

    for (q_range, sub_out) in partials {
        let sub_q_width = q_range.len() * head_dim;
        for m_idx in 0..m {
            let dst_off = m_idx * full_q_width + q_range.start * head_dim;
            let src_off = m_idx * sub_q_width;
            out[dst_off..dst_off + sub_q_width]
                .copy_from_slice(&sub_out[src_off..src_off + sub_q_width]);
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "scalar GQA validator checks all dimensions explicitly"
)]
fn validate_gqa_prefill_scalar(
    out: &[f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
    scores_scratch: &[f32],
) -> Result<(), K3Error> {
    if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || m == 0 || total_seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "phi3".to_owned(),
            reason: "GQA prefill dimensions must be non-zero".to_owned(),
        });
    }
    if n_heads_q % n_heads_kv != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "phi3".to_owned(),
            reason: format!("n_heads_q ({n_heads_q}) not divisible by n_heads_kv ({n_heads_kv})"),
        });
    }
    if !scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "phi3".to_owned(),
            reason: format!("attention scale must be finite (got {scale})"),
        });
    }
    if sliding_window == Some(0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "phi3".to_owned(),
            reason: "sliding_window must be >= 1 when set".to_owned(),
        });
    }
    let query_end = start_pos
        .checked_add(m)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_prefill_start_pos".to_owned(),
            offset: start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: "start_pos + m overflow".to_owned(),
        })?;
    if query_end > total_seq_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_prefill_start_pos".to_owned(),
            offset: start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: format!("start_pos + m ({query_end}) exceeds total_seq_len ({total_seq_len})"),
        });
    }
    let q_len = m * n_heads_q * head_dim;
    let kv_width = n_heads_kv * head_dim;
    let kv_min = total_seq_len * kv_width;
    let min_scratch = sliding_window.map_or(query_end, |w| w.min(query_end));
    if q.len() != q_len
        || out.len() != q_len
        || k_cache.len() < kv_min
        || v_cache.len() < kv_min
        || scores_scratch.len() < min_scratch
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_causal_attention_prefill".to_owned(),
            offset: 0,
            byte_len: size_of_val(q),
            buffer_len: size_of_val(k_cache),
            reason: "GQA prefill buffer or scratch dimension mismatch".to_owned(),
        });
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_f32 kernel ABI directly"
)]
fn gqa_prefill_causal_scalar(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    validate_gqa_prefill_scalar(
        out,
        q,
        k_cache,
        v_cache,
        m,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_seq_len,
        start_pos,
        sliding_window,
        scale,
        scores_scratch,
    )?;

    let group_size = n_heads_q / n_heads_kv;
    for m_idx in 0..m {
        let query_pos = start_pos + m_idx;
        let valid_end = (query_pos + 1).min(total_seq_len);
        let valid_start =
            sliding_window.map_or(0, |w| query_pos.saturating_add(1).saturating_sub(w));
        let active_len = valid_end.saturating_sub(valid_start);
        let score_buf = &mut scores_scratch[..active_len];

        for h_q in 0..n_heads_q {
            let h_kv = h_q / group_size;
            let q_head = &q[(m_idx * n_heads_q + h_q) * head_dim..][..head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for (slot, s) in score_buf.iter_mut().zip(valid_start..valid_end) {
                let kv_off = (s * n_heads_kv + h_kv) * head_dim;
                let sc = dot_product_f32(q_head, &k_cache[kv_off..kv_off + head_dim]) * scale;
                *slot = sc;
                if sc > max_score {
                    max_score = sc;
                }
            }
            let mut sum_exp = 0.0_f32;
            for slot in &mut score_buf[..active_len] {
                let e = (*slot - max_score).exp();
                *slot = e;
                sum_exp += e;
            }
            #[expect(clippy::cast_precision_loss, reason = "active_len fits cleanly in f32")]
            let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
                1.0_f32 / sum_exp
            } else {
                1.0_f32 / (active_len.max(1) as f32)
            };
            let out_head = &mut out[(m_idx * n_heads_q + h_q) * head_dim..][..head_dim];
            out_head.fill(0.0);
            for (s, &e) in (valid_start..valid_end).zip(&score_buf[..active_len]) {
                let w = e * inv_sum;
                let kv_off = (s * n_heads_kv + h_kv) * head_dim;
                let v_head = &v_cache[kv_off..kv_off + head_dim];
                for (dst, &v_val) in out_head.iter_mut().zip(v_head) {
                    *dst += w * v_val;
                }
            }
        }
    }
    Ok(())
}

/// Computes batched causal GQA prefill over query rows `[start_pos..start_pos + m)` attending
/// over KV history `[0..total_seq_len)`.
///
/// Dispatches across active `WorkerPool` threads partitioned by KV heads (`with-kernels`) or
/// portable scalar math (`--no-default-features`).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on invalid
/// geometry or buffer dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_f32 kernel ABI directly"
)]
pub(crate) fn gqa_prefill_causal_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let config =
            AttentionMaskConfig::causal(start_pos, scale).with_sliding_window(sliding_window);
        #[cfg(not(miri))]
        {
            let workers = rayon::current_num_threads().min(n_heads_kv);
            if workers > 1 && n_heads_kv >= 2 {
                return gqa_prefill_causal_parallel(
                    out,
                    q,
                    k_cache,
                    v_cache,
                    m,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    total_seq_len,
                    sliding_window,
                    config,
                    workers,
                );
            }
        }

        gqa_causal_attention_prefill_f32(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            config,
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gqa_prefill_causal_scalar(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            start_pos,
            sliding_window,
            scale,
            scores_scratch,
        )
    }
}
