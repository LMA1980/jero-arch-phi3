//! Quantized block dequantization, geometry validation, and projection kernels for Phi-3 / Phi-4.
//!
//! Dispatches to `jero-kernels-rvv` hardware/scalar kernels when `with-kernels` is enabled, and
//! executes bit-equivalent portable scalar math across `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`,
//! `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` when built with `--no-default-features`.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
    Q4_K_M_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, VectorKernel, bf16_to_f32_finite,
    dequantize_token_row, f16_to_f32_finite, gqa_causal_attention_f32, swiglu_f32, vec_dot_bf16,
    vec_dot_f16, vec_dot_q4_0,
};

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

/// Decodes a single 1D normalization tensor (`F32`, `F16`, or `BF16`) into `expected_len` floats.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if the element count or byte length mismatches, or
/// [`K3Error::CapabilityMismatch`] if `tensor.quant()` is not an unquantized norm format.
pub(crate) fn decode_1d_norm(
    tensor: &TensorView<'_>,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if tensor.num_elements() != expected_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "norm element count {} != expected hidden_dim {expected_len}",
                tensor.num_elements()
            ),
        });
    }
    let mut out = vec![0.0_f32; expected_len];
    decode_unquant_payload(tensor, tensor.as_bytes(), &mut out)?;
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

/// Computes `output[0..expected_rows] = weights * input[0..expected_cols]` across all
/// admitted Phi-3 / Phi-4 quantization types (`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`,
/// `Q4_K_M`, `Q5_K`, `Q6_K`) without silent synthetic fallbacks.
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
        project_matvec_kernels(output, weights, rows, cols, input)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        project_matvec_scalar(output, weights, rows, cols, input)
    }
}

#[cfg(feature = "with-kernels")]
fn project_matvec_kernels(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(output, weights, input),
        QuantType::Q8_0 => BaselineKernel::new().vec_dot_q8_0(output, weights, input),
        QuantType::Q6_K => BaselineKernel::new().vec_dot_q6_k(output, weights, input),
        QuantType::Q5_K => BaselineKernel::new().vec_dot_q5_k(output, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(output, weights, input),
        QuantType::F16 => vec_dot_f16(output, weights, input),
        QuantType::BF16 => vec_dot_bf16(output, weights, input),
        QuantType::Q5_0 => dot_block_rows::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_0_block,
        ),
        QuantType::F32 => project_f32_rows(output, weights, rows, cols, input),
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
        QuantType::F32 => project_f32_rows(output, weights, rows, cols, input),
        QuantType::F16 | QuantType::BF16 => {
            let mut row_buf = vec![0.0_f32; cols];
            let stride = cols * 2;
            let raw = weights.as_bytes();
            for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
                decode_unquant_payload(weights, &raw[r * stride..(r + 1) * stride], &mut row_buf)?;
                *out_slot = dot_product_f32(&row_buf, input);
            }
            Ok(())
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

fn project_f32_rows(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    let raw = weights.as_bytes();
    let stride = cols * 4;
    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
        decode_unquant_payload(weights, &raw[r * stride..(r + 1) * stride], &mut row_buf)?;
        *out_slot = dot_product_f32(&row_buf, input);
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

/// Computes causal Grouped-Query Attention across `[0..seq_len)` cached KV slots for one query token.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on invalid
/// geometry or buffer dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_f32 kernel ABI directly"
)]
pub(crate) fn gqa_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_f32(
            out, q, k_cache, v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, scale,
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
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "gqa_causal_attention".to_owned(),
                offset: 0,
                byte_len: size_of_val(q),
                buffer_len: size_of_val(k_cache),
                reason: "GQA query or KV cache buffer dimension mismatch".to_owned(),
            });
        }
        let group_size = n_heads_q / n_heads_kv;
        let mut scores = vec![0.0_f32; seq_len];
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
            for sc in &mut scores[..seq_len] {
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
