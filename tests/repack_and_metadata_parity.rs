//! Integration tests for load-time pre-splitting of fused `attn_qkv` and `ffn_up` tensors into
//! `128-byte`-aligned `RepackedWeight` sub-matrices, cached 1D norm vectors, multi-quant
//! projection parity, and full-scale `Phi-3.5` / `Phi-4-mini` / `Phi-4` stop-token & vocab
//! bounds contracts.

use std::borrow::Cow;

use jero_arch_phi3::{
    PHI4_14B_VOCAB_SIZE, PHI4_MINI_VOCAB_SIZE, PHI35_PADDED_VOCAB_SIZE, Phi3LayerRepack,
    Phi3Recipe, Phi3SplitMatrix, PhiChatDialect, REPACK_TILE_ROWS, RopeScalingConfig,
};
use jero_core::gguf::{
    GgufArray, GgufFile, GgufValue, test_helpers::create_synthetic_gguf_v3_with_kvs,
};
use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, StopReason, TensorView,
};

struct BenchWeights<'a> {
    tensors: Vec<(String, TensorView<'a>)>,
}

impl ModelWeights for BenchWeights<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, view)| *view)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor not found".to_owned(),
            })
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.iter().map(|(_, v)| v.byte_len()).sum()
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0);
    let mant = bits & 0x007F_FFFF;
    if val == 0.0 {
        return sign;
    }
    let f16_exp = (exp - 127) + 15;
    if f16_exp <= 0 {
        return sign;
    }
    if f16_exp >= 31 {
        return sign | 0x7C00;
    }
    let exp_u16 = u16::try_from(f16_exp).unwrap_or(0) << 10;
    let mant_u16 = u16::try_from(mant >> 13).unwrap_or(0);
    sign | exp_u16 | mant_u16
}

fn f16_to_f32(bits: u16) -> f32 {
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
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

fn bf16_to_f32(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

fn pattern_element_f32(r: usize, c: usize, seed: usize) -> f32 {
    let bucket = u8::try_from((r * 13 + c * 7 + seed) % 31).expect("modulus fits u8");
    (f32::from(bucket) - 15.0) * 0.025
}

fn encode_k_quant_blocks(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::new();
    match quant {
        QuantType::Q4_K_M => {
            for r in 0..rows {
                for b in 0..(cols / 256) {
                    let mut blk = vec![0_u8; 144];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(0.025).to_le_bytes());
                    blk[2..4].copy_from_slice(&f32_to_f16_bits(0.012).to_le_bytes());
                    for (i, byte) in blk[4..16].iter_mut().enumerate() {
                        *byte = u8::try_from(((r + b + i + seed) % 19) + 4).unwrap_or(5);
                    }
                    for (i, byte) in blk[16..144].iter_mut().enumerate() {
                        *byte = u8::try_from((r * 7 + b + i * 3 + seed) & 0xFF).unwrap_or(0);
                    }
                    out.extend_from_slice(&blk);
                }
            }
        }
        QuantType::Q5_K => {
            for r in 0..rows {
                for b in 0..(cols / 256) {
                    let mut blk = vec![0_u8; 176];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                    blk[2..4].copy_from_slice(&f32_to_f16_bits(0.01).to_le_bytes());
                    for (i, byte) in blk[4..16].iter_mut().enumerate() {
                        *byte = u8::try_from(((r + b + i + seed) % 17) + 4).unwrap_or(5);
                    }
                    for (i, byte) in blk[16..48].iter_mut().enumerate() {
                        *byte = u8::try_from((r * 3 + b + i * 7 + seed) & 0xFF).unwrap_or(0);
                    }
                    for (i, byte) in blk[48..176].iter_mut().enumerate() {
                        *byte = u8::try_from((r * 5 + b + i * 11 + seed) & 0xFF).unwrap_or(0);
                    }
                    out.extend_from_slice(&blk);
                }
            }
        }
        QuantType::Q6_K => {
            for r in 0..rows {
                for b in 0..(cols / 256) {
                    let mut blk = vec![0_u8; 210];
                    for (i, byte) in blk[0..128].iter_mut().enumerate() {
                        *byte = u8::try_from((r * 3 + b + i * 5 + seed) & 0xFF).unwrap_or(0);
                    }
                    for (i, byte) in blk[128..192].iter_mut().enumerate() {
                        *byte = u8::try_from((r * 7 + b + i * 3 + seed) & 0xFF).unwrap_or(0);
                    }
                    for (i, byte) in blk[192..208].iter_mut().enumerate() {
                        let signed = i8::try_from((r + b + i + seed) % 7).unwrap_or(0) - 3;
                        *byte = signed.to_ne_bytes()[0];
                    }
                    blk[208..210].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                    out.extend_from_slice(&blk);
                }
            }
        }
        other => panic!("unexpected non-K quant: {other:?}"),
    }
    out
}

fn encode_matrix(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::new();
    match quant {
        QuantType::F32 => {
            for r in 0..rows {
                for c in 0..cols {
                    out.extend_from_slice(&pattern_element_f32(r, c, seed).to_le_bytes());
                }
            }
        }
        QuantType::F16 => {
            for r in 0..rows {
                for c in 0..cols {
                    let v = pattern_element_f32(r, c, seed);
                    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            }
        }
        QuantType::BF16 => {
            for r in 0..rows {
                for c in 0..cols {
                    let v = pattern_element_f32(r, c, seed);
                    let bits = u16::try_from(v.to_bits() >> 16).unwrap_or(0);
                    out.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
        QuantType::Q4_0 => {
            for r in 0..rows {
                for b in 0..(cols / 32) {
                    out.extend_from_slice(&f32_to_f16_bits(0.03).to_le_bytes());
                    for j in 0..16 {
                        let q0 = u8::try_from((r * 5 + b * 3 + j + seed) % 16).unwrap_or(8);
                        let q1 = u8::try_from((r * 7 + b * 5 + j + 4 + seed) % 16).unwrap_or(8);
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
            }
        }
        QuantType::Q5_0 => {
            for r in 0..rows {
                for b in 0..(cols / 32) {
                    out.extend_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                    let qh = u32::try_from(r.wrapping_mul(0x0101_0101) ^ b.wrapping_mul(0x1357))
                        .unwrap_or(0);
                    out.extend_from_slice(&qh.to_le_bytes());
                    for j in 0..16 {
                        let q0 = u8::try_from((r * 3 + b + j + seed) % 16).unwrap_or(0);
                        let q1 = u8::try_from((r * 11 + b + j + seed) % 16).unwrap_or(0);
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
            }
        }
        QuantType::Q8_0 => {
            for r in 0..rows {
                for b in 0..(cols / 32) {
                    out.extend_from_slice(&f32_to_f16_bits(0.015).to_le_bytes());
                    for j in 0..32 {
                        let qi = i8::try_from((r * 9 + b * 5 + j + seed) % 23).unwrap_or(0) - 11;
                        out.push(qi.to_ne_bytes()[0]);
                    }
                }
            }
        }
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            return encode_k_quant_blocks(quant, cols, rows, seed);
        }
        other => panic!("unsupported quant in test: {other:?}"),
    }
    out
}

#[expect(
    clippy::too_many_lines,
    reason = "standalone reference decoder across all 9 admitted Phi-3/Phi-4 quantization formats"
)]
fn ref_dequant_tensor(quant: QuantType, raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    let unpack_6bit = |scales: &[u8]| -> ([u8; 8], [u8; 8]) {
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
    };

    match quant {
        QuantType::F32 => {
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        QuantType::F16 => {
            for c in raw.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        QuantType::BF16 => {
            for c in raw.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        QuantType::Q4_0 => {
            for blk in raw.chunks_exact(18) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let mut row = [0.0_f32; 32];
                for (j, &p) in blk[2..18].iter().enumerate() {
                    row[j] = d * (f32::from(p & 0x0F) - 8.0);
                    row[j + 16] = d * (f32::from((p >> 4) & 0x0F) - 8.0);
                }
                out.extend_from_slice(&row);
            }
        }
        QuantType::Q5_0 => {
            for blk in raw.chunks_exact(22) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let mut row = [0.0_f32; 32];
                for (j, &p) in blk[6..22].iter().enumerate() {
                    let h0 = u8::try_from(((qh >> j) & 1) << 4).unwrap_or(0);
                    let h1 = u8::try_from(((qh >> (j + 16)) & 1) << 4).unwrap_or(0);
                    row[j] = d * (f32::from((p & 0x0F) | h0) - 16.0);
                    row[j + 16] = d * (f32::from(((p >> 4) & 0x0F) | h1) - 16.0);
                }
                out.extend_from_slice(&row);
            }
        }
        QuantType::Q8_0 => {
            for blk in raw.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    #[expect(clippy::cast_possible_wrap, reason = "Q8_0 signed i8 weight")]
                    let qi = q as i8;
                    out.push(d * f32::from(qi));
                }
            }
        }
        QuantType::Q4_K_M => {
            for blk in raw.chunks_exact(144) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let (sc, m) = unpack_6bit(&blk[4..16]);
                let mut buf = [0.0_f32; 256];
                for c in 0..4 {
                    let d1 = d * f32::from(sc[c * 2]);
                    let m1 = dmin * f32::from(m[c * 2]);
                    let d2 = d * f32::from(sc[c * 2 + 1]);
                    let m2 = dmin * f32::from(m[c * 2 + 1]);
                    for l in 0..32 {
                        buf[c * 64 + l] = d1 * f32::from(blk[16 + c * 32 + l] & 0x0F) - m1;
                        buf[c * 64 + 32 + l] =
                            d2 * f32::from((blk[16 + c * 32 + l] >> 4) & 0x0F) - m2;
                    }
                }
                out.extend_from_slice(&buf);
            }
        }
        QuantType::Q5_K => {
            for blk in raw.chunks_exact(176) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let (sc, m) = unpack_6bit(&blk[4..16]);
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let mut buf = [0.0_f32; 256];
                let mut u1 = 1_u8;
                let mut u2 = 2_u8;
                for c in 0..4 {
                    let d1 = d * f32::from(sc[c * 2]);
                    let m1 = dmin * f32::from(m[c * 2]);
                    let d2 = d * f32::from(sc[c * 2 + 1]);
                    let m2 = dmin * f32::from(m[c * 2 + 1]);
                    for l in 0..32 {
                        let hb1 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                        let hb2 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                        buf[c * 64 + l] = d1 * f32::from((qs[c * 32 + l] & 0x0F) + hb1) - m1;
                        buf[c * 64 + 32 + l] =
                            d2 * f32::from(((qs[c * 32 + l] >> 4) & 0x0F) + hb2) - m2;
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
                out.extend_from_slice(&buf);
            }
        }
        QuantType::Q6_K => {
            for blk in raw.chunks_exact(210) {
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208];
                let mut buf = [0.0_f32; 256];
                for (h_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
                    let lb = h_idx * 64;
                    let hb = h_idx * 32;
                    let sb = h_idx * 8;
                    for l in 0..32 {
                        let is = l / 16;
                        #[expect(clippy::cast_possible_wrap, reason = "6-bit signed")]
                        let q1 = (((ql[lb + l] & 0x0F) | ((qh[hb + l] & 0x03) << 4)) as i8) - 32;
                        #[expect(clippy::cast_possible_wrap, reason = "6-bit signed")]
                        let q2 = (((ql[lb + l + 32] & 0x0F) | (((qh[hb + l] >> 2) & 0x03) << 4))
                            as i8)
                            - 32;
                        #[expect(clippy::cast_possible_wrap, reason = "6-bit signed")]
                        let q3 = ((((ql[lb + l] >> 4) & 0x0F) | (((qh[hb + l] >> 4) & 0x03) << 4))
                            as i8)
                            - 32;
                        #[expect(clippy::cast_possible_wrap, reason = "6-bit signed")]
                        let q4 = ((((ql[lb + l + 32] >> 4) & 0x0F)
                            | (((qh[hb + l] >> 6) & 0x03) << 4))
                            as i8)
                            - 32;
                        #[expect(clippy::cast_possible_wrap, reason = "8-bit signed scale")]
                        let s1 = f32::from(scales[sb + is] as i8);
                        #[expect(clippy::cast_possible_wrap, reason = "8-bit signed scale")]
                        let s2 = f32::from(scales[sb + is + 2] as i8);
                        #[expect(clippy::cast_possible_wrap, reason = "8-bit signed scale")]
                        let s3 = f32::from(scales[sb + is + 4] as i8);
                        #[expect(clippy::cast_possible_wrap, reason = "8-bit signed scale")]
                        let s4 = f32::from(scales[sb + is + 6] as i8);
                        buf[n + l] = d * s1 * f32::from(q1);
                        buf[n + l + 32] = d * s2 * f32::from(q2);
                        buf[n + l + 64] = d * s3 * f32::from(q3);
                        buf[n + l + 96] = d * s4 * f32::from(q4);
                    }
                }
                out.extend_from_slice(&buf);
            }
        }
        other => panic!("unexpected quant {other:?}"),
    }
    out
}

fn ref_matvec_slice(
    dequant: &[f32],
    cols: usize,
    row_start: usize,
    row_count: usize,
    input: &[f32],
) -> Vec<f32> {
    (row_start..row_start + row_count)
        .map(|r| {
            dequant[r * cols..(r + 1) * cols]
                .iter()
                .zip(input)
                .map(|(&w, &x)| w * x)
                .sum()
        })
        .collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn argmax_idx(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(idx, _)| idx)
}

fn verify_split_subset(
    geom_name: &str,
    quant: QuantType,
    raw_ref: &[f32],
    input: &[f32],
    cols: usize,
    checks: &[(&Phi3SplitMatrix, usize, usize)],
) {
    let mut deq_row = vec![0.0_f32; cols];
    for &(split, offset, count) in checks {
        assert_eq!(split.rows(), count);
        assert_eq!(split.cols(), cols);
        let expected_tiles = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
            0
        } else {
            count.div_ceil(REPACK_TILE_ROWS)
        };
        assert_eq!(split.n_tiles(), expected_tiles);
        if expected_tiles > 0 {
            assert!(split.is_128_byte_aligned());
        }

        for r in 0..count {
            split.dequantize_row(r, &mut deq_row).expect("dequant row");
            let expected = &raw_ref[(offset + r) * cols..(offset + r + 1) * cols];
            let max_err = deq_row
                .iter()
                .zip(expected)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                max_err <= 2e-5,
                "{geom_name}/{quant:?} {} row {r} max_err={max_err}",
                split.name()
            );
        }

        let mut actual_mv = vec![0.0_f32; count];
        split.gemv(&mut actual_mv, input).expect("split gemv");
        let expected_mv = ref_matvec_slice(raw_ref, cols, offset, count, input);
        let similarity = cosine_sim(&actual_mv, &expected_mv);
        assert!(
            similarity > 0.998,
            "{geom_name}/{quant:?} {} gemv cosine={similarity}",
            split.name()
        );
    }
}

fn verify_presplit_qkv_and_ffn(
    geom_name: &str,
    hq: usize,
    hkv: usize,
    hd: usize,
    ffn_dim: usize,
    quant: QuantType,
    input: &[f32],
) {
    let cols = input.len();
    let q_dim = hq * hd;
    let kv_dim = hkv * hd;
    let qkv_rows = q_dim + 2 * kv_dim;
    let up_rows = 2 * ffn_dim;

    let qkv_raw = encode_matrix(quant, cols, qkv_rows, 11);
    let qkv_ref = ref_dequant_tensor(quant, &qkv_raw);
    let qkv_shape = [cols, qkv_rows];
    let qkv_view = TensorView::from_bytes("blk.0.attn_qkv.weight", quant, &qkv_shape, 0, &qkv_raw)
        .expect("qkv view");

    let wq = Phi3SplitMatrix::from_tensor_slice(&qkv_view, "blk.0.attn_q.weight", 0, q_dim, cols)
        .expect("split wq");
    let wk =
        Phi3SplitMatrix::from_tensor_slice(&qkv_view, "blk.0.attn_k.weight", q_dim, kv_dim, cols)
            .expect("split wk");
    let wv = Phi3SplitMatrix::from_tensor_slice(
        &qkv_view,
        "blk.0.attn_v.weight",
        q_dim + kv_dim,
        kv_dim,
        cols,
    )
    .expect("split wv");
    verify_split_subset(
        geom_name,
        quant,
        &qkv_ref,
        input,
        cols,
        &[
            (&wq, 0, q_dim),
            (&wk, q_dim, kv_dim),
            (&wv, q_dim + kv_dim, kv_dim),
        ],
    );

    let up_raw = encode_matrix(quant, cols, up_rows, 37);
    let up_ref = ref_dequant_tensor(quant, &up_raw);
    let up_shape = [cols, up_rows];
    let up_view = TensorView::from_bytes("blk.0.ffn_up.weight", quant, &up_shape, 0, &up_raw)
        .expect("up view");

    let wgate =
        Phi3SplitMatrix::from_tensor_slice(&up_view, "blk.0.ffn_gate.weight", 0, ffn_dim, cols)
            .expect("split wgate");
    let wup =
        Phi3SplitMatrix::from_tensor_slice(&up_view, "blk.0.ffn_up.weight", ffn_dim, ffn_dim, cols)
            .expect("split wup");
    verify_split_subset(
        geom_name,
        quant,
        &up_ref,
        input,
        cols,
        &[(&wgate, 0, ffn_dim), (&wup, ffn_dim, ffn_dim)],
    );
}

#[test]
fn test_all_admitted_quants_presplit_dequant_and_projection_parity() {
    const HIDDEN: usize = 256;
    let geometries: &[(&str, usize, usize, usize, usize)] = &[
        ("MHA_32_32", 32, 32, 2, 16), // q=64, k=64, v=64, ffn=16
        ("GQA_32_8", 32, 8, 2, 12),   // q=64, k=16, v=16, ffn=12 (non-mult-of-8!)
        ("GQA_40_10", 40, 10, 2, 14), // q=80, k=20, v=20 (20%8==4), ffn=14 (14%8==6)
    ];
    let quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q5_0,
        QuantType::Q8_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
    ];
    let input: Vec<f32> = (0..HIDDEN)
        .map(|i| {
            let bucket = u8::try_from((i * 7 + 5) % 29).expect("modulus fits u8");
            (f32::from(bucket) - 14.0) * 0.08
        })
        .collect();

    for &(geom_name, hq, hkv, hd, ffn_dim) in geometries {
        for &quant in &quants {
            verify_presplit_qkv_and_ffn(geom_name, hq, hkv, hd, ffn_dim, quant, &input);
        }
    }
}

fn build_mixed_quant_2layer_specs(
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    vocab: usize,
) -> Vec<(String, QuantType, Vec<usize>, Vec<u8>)> {
    let mut bufs = vec![
        (
            "token_embd.weight".to_owned(),
            QuantType::Q4_0,
            vec![hidden, vocab],
            encode_matrix(QuantType::Q4_0, hidden, vocab, 3),
        ),
        (
            "output_norm.weight".to_owned(),
            QuantType::F16,
            vec![hidden],
            encode_matrix(QuantType::F16, hidden, 1, 9),
        ),
        (
            "output.weight".to_owned(),
            QuantType::Q8_0,
            vec![hidden, vocab],
            encode_matrix(QuantType::Q8_0, hidden, vocab, 17),
        ),
    ];

    let layer_quants = [
        (
            QuantType::Q5_K,
            QuantType::Q4_K_M,
            QuantType::Q6_K,
            QuantType::F16,
        ),
        (
            QuantType::Q4_0,
            QuantType::Q5_0,
            QuantType::Q8_0,
            QuantType::BF16,
        ),
    ];
    for (layer, &(qkv_q, up_q, down_q, norm_q)) in layer_quants.iter().enumerate() {
        bufs.push((
            format!("blk.{layer}.attn_norm.weight"),
            norm_q,
            vec![hidden],
            encode_matrix(norm_q, hidden, 1, 21 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.attn_qkv.weight"),
            qkv_q,
            vec![hidden, q_dim + 2 * kv_dim],
            encode_matrix(qkv_q, hidden, q_dim + 2 * kv_dim, 31 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.attn_output.weight"),
            QuantType::Q8_0,
            vec![q_dim, hidden],
            encode_matrix(QuantType::Q8_0, q_dim, hidden, 41 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_norm.weight"),
            QuantType::BF16,
            vec![hidden],
            encode_matrix(QuantType::BF16, hidden, 1, 51 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_up.weight"),
            up_q,
            vec![hidden, 2 * ffn_dim],
            encode_matrix(up_q, hidden, 2 * ffn_dim, 61 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_down.weight"),
            down_q,
            vec![ffn_dim, hidden],
            encode_matrix(down_q, ffn_dim, hidden, 71 + layer),
        ));
    }
    bufs
}

#[test]
fn test_end_to_end_repack_prefill_decode_cosine_and_top1_parity() {
    const HIDDEN: usize = 256;
    const HEADS_Q: usize = 40;
    const HEADS_KV: usize = 10;
    const HEAD_DIM: usize = 8; // q_dim = 320, kv_dim = 80
    const Q_DIM: usize = HEADS_Q * HEAD_DIM;
    const KV_DIM: usize = HEADS_KV * HEAD_DIM;
    const FFN_DIM: usize = 256;
    const VOCAB: usize = 16;

    let mut recipe = Phi3Recipe::new_phi4();
    recipe.num_layers = 2;
    recipe.hidden_dim = HIDDEN;
    recipe.ffn_dim = FFN_DIM;
    recipe.n_heads_q = HEADS_Q;
    recipe.n_heads_kv = HEADS_KV;
    recipe.head_dim = HEAD_DIM;
    recipe.rotary_dim = Some(HEAD_DIM);
    recipe.vocab_size = VOCAB;
    recipe.rope_scaling = RopeScalingConfig::Linear { factor: 1.5 };

    let bufs = build_mixed_quant_2layer_specs(HIDDEN, Q_DIM, KV_DIM, FFN_DIM, VOCAB);
    let views: Vec<(String, TensorView<'_>)> = bufs
        .iter()
        .map(|(name, quant, shape, raw)| {
            (
                name.clone(),
                TensorView::from_bytes(name, *quant, shape, 0, raw).expect("view"),
            )
        })
        .collect();
    let weights = BenchWeights { tensors: views };

    let cache = recipe.build_repack_cache(&weights).expect("repack cache");
    assert_eq!(cache.layers().len(), 2);
    for layer in 0..2 {
        let lr = Phi3LayerRepack::from_weights(&recipe, layer, &weights).expect("layer repack");
        assert_eq!(lr.wq().rows(), Q_DIM);
        assert_eq!(lr.wk().rows(), KV_DIM);
        assert_eq!(lr.wv().rows(), KV_DIM);
    }

    let mut logits_direct = [0.0_f32; VOCAB];
    let mut kv_direct = vec![0.0_f32; 2 * 8 * KV_DIM * 2];
    let mut ctx_d =
        ExecutionContext::new(&[1, 4, 7], 8, &mut logits_direct, &mut kv_direct).expect("d");
    recipe
        .prefill_step(&mut ctx_d, &weights)
        .expect("d prefill");

    let mut logits_repack = [0.0_f32; VOCAB];
    let mut kv_repack = vec![0.0_f32; 2 * 8 * KV_DIM * 2];
    let mut ctx_r =
        ExecutionContext::new(&[1, 4, 7], 8, &mut logits_repack, &mut kv_repack).expect("r");
    recipe
        .prefill_step_with_repack(&mut ctx_r, &weights, &cache)
        .expect("r prefill");

    let prefill_cos = cosine_sim(ctx_r.logits(), ctx_d.logits());
    assert!(prefill_cos > 0.999, "prefill cosine={prefill_cos}");
    assert_eq!(argmax_idx(ctx_r.logits()), argmax_idx(ctx_d.logits()));

    ctx_d.prepare_decode(9).expect("d prep decode");
    recipe.decode_step(&mut ctx_d, &weights).expect("d decode");

    ctx_r.prepare_decode(9).expect("r prep decode");
    recipe
        .decode_step_with_repack(&mut ctx_r, &weights, &cache)
        .expect("r decode");

    let decode_cos = cosine_sim(&logits_repack, &logits_direct);
    assert!(decode_cos > 0.999, "decode cosine={decode_cos}");
    assert_eq!(argmax_idx(&logits_repack), argmax_idx(&logits_direct));
}

#[test]
fn test_full_scale_phi35_vocab_and_stop_contracts() {
    let mut phi35_vocab: Vec<Cow<'_, str>> = (0..PHI35_PADDED_VOCAB_SIZE)
        .map(|i| Cow::Owned(format!("t{i}")))
        .collect();
    phi35_vocab[1] = Cow::Borrowed("<s>");
    phi35_vocab[32_000] = Cow::Borrowed("<|endoftext|>");
    phi35_vocab[32_007] = Cow::Borrowed("<|end|>");
    let phi35_refs: Vec<&str> = phi35_vocab.iter().map(Cow::as_ref).collect();
    let phi35_vocab_u32 = u32::try_from(PHI35_PADDED_VOCAB_SIZE).expect("fits u32");
    let phi35_kvs = [
        ("phi3.embedding_length", GgufValue::Uint32(2)),
        ("phi3.block_count", GgufValue::Uint32(1)),
        ("phi3.attention.head_count", GgufValue::Uint32(1)),
        ("phi3.vocab_size", GgufValue::Uint32(phi35_vocab_u32)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(phi35_refs)),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(32_007)),
    ];
    let phi35_bytes = create_synthetic_gguf_v3_with_kvs(
        "phi3",
        32,
        &phi35_kvs,
        &[
            (
                "token_embd.weight",
                QuantType::F16,
                &[2, PHI35_PADDED_VOCAB_SIZE],
            ),
            (
                "output.weight",
                QuantType::F16,
                &[2, PHI35_PADDED_VOCAB_SIZE],
            ),
        ],
    );
    let phi35_gguf = GgufFile::parse(&phi35_bytes).expect("parse phi3.5");
    let phi35_recipe = Phi3Recipe::from_gguf(&phi35_gguf).expect("phi3.5 recipe");
    let stop35 = phi35_recipe
        .extract_stop_metadata(&phi35_gguf)
        .expect("phi3.5 stop");
    assert_eq!(stop35.dialect, PhiChatDialect::Phi3End);
    assert_eq!(stop35.vocab_size, PHI35_PADDED_VOCAB_SIZE);
    assert_eq!(stop35.end_token_id, Some(32_007));
    assert!(!stop35.has_tied_lm_head);
    let logits35 = vec![0.0_f32; PHI35_PADDED_VOCAB_SIZE];
    stop35
        .validate_logits_slice("phi3", &logits35)
        .expect("valid 32,064 logits");
    assert!(matches!(
        stop35.validate_logits_slice("phi3", &vec![0.0_f32; 32_000]),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    let crit35 = stop35.stop_criteria(32, 256);
    assert_eq!(crit35.evaluate(42, 1, 2), None);
    assert_eq!(
        crit35.evaluate(32_007, 2, 3),
        Some(StopReason::EosToken(32_007))
    );
}

#[test]
fn test_full_scale_phi4_mini_vocab_and_stop_contracts() {
    let mut phi4m_vocab: Vec<Cow<'_, str>> = (0..PHI4_MINI_VOCAB_SIZE)
        .map(|i| Cow::Owned(format!("m{i}")))
        .collect();
    phi4m_vocab[199_999] = Cow::Borrowed("<|endoftext|>");
    phi4m_vocab[200_020] = Cow::Borrowed("<|end|>");
    let phi4m_refs: Vec<&str> = phi4m_vocab.iter().map(Cow::as_ref).collect();
    let phi4m_vocab_u32 = u32::try_from(PHI4_MINI_VOCAB_SIZE).expect("fits u32");
    let phi4m_kvs = [
        ("phi3.embedding_length", GgufValue::Uint32(2)),
        ("phi3.block_count", GgufValue::Uint32(1)),
        ("phi3.attention.head_count", GgufValue::Uint32(1)),
        ("phi3.vocab_size", GgufValue::Uint32(phi4m_vocab_u32)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(phi4m_refs)),
        ),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(200_020)),
    ];
    let phi4m_bytes = create_synthetic_gguf_v3_with_kvs(
        "phi3",
        32,
        &phi4m_kvs,
        &[(
            "token_embd.weight",
            QuantType::F16,
            &[2, PHI4_MINI_VOCAB_SIZE],
        )],
    );
    let phi4m_gguf = GgufFile::parse(&phi4m_bytes).expect("parse phi4-mini");
    let phi4m_recipe = Phi3Recipe::from_gguf(&phi4m_gguf).expect("phi4-mini recipe");
    let stop4m = phi4m_recipe
        .extract_stop_metadata(&phi4m_gguf)
        .expect("phi4-mini stop");
    assert_eq!(stop4m.vocab_size, PHI4_MINI_VOCAB_SIZE);
    assert!(stop4m.has_tied_lm_head);
    assert_eq!(stop4m.end_token_id, Some(200_020));
    stop4m
        .validate_logits_slice("phi3", &vec![0.0_f32; PHI4_MINI_VOCAB_SIZE])
        .expect("valid 200,064 logits");
    assert!(matches!(
        stop4m.validate_logits_slice("phi3", &vec![0.0_f32; PHI4_MINI_VOCAB_SIZE - 64]),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_full_scale_phi4_14b_vocab_and_stop_contracts() {
    let mut phi4_vocab: Vec<Cow<'_, str>> = (0..PHI4_14B_VOCAB_SIZE)
        .map(|i| Cow::Owned(format!("p{i}")))
        .collect();
    phi4_vocab[100_257] = Cow::Borrowed("<|endoftext|>");
    phi4_vocab[100_265] = Cow::Borrowed("<|im_end|>");
    phi4_vocab[100_266] = Cow::Borrowed("<|im_sep|>");
    let phi4_refs: Vec<&str> = phi4_vocab.iter().map(Cow::as_ref).collect();
    let phi4_vocab_u32 = u32::try_from(PHI4_14B_VOCAB_SIZE).expect("fits u32");
    let phi4_kvs = [
        ("phi4.embedding_length", GgufValue::Uint32(2)),
        ("phi4.block_count", GgufValue::Uint32(1)),
        ("phi4.attention.head_count", GgufValue::Uint32(1)),
        ("phi4.vocab_size", GgufValue::Uint32(phi4_vocab_u32)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(phi4_refs)),
        ),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(100_265)),
    ];
    let phi4_bytes = create_synthetic_gguf_v3_with_kvs(
        "phi4",
        32,
        &phi4_kvs,
        &[("output.weight", QuantType::F16, &[2, PHI4_14B_VOCAB_SIZE])],
    );
    let phi4_gguf = GgufFile::parse(&phi4_bytes).expect("parse phi4-14b");
    let phi4_recipe = Phi3Recipe::from_gguf(&phi4_gguf).expect("phi4-14b recipe");
    let stop4 = phi4_recipe
        .extract_stop_metadata(&phi4_gguf)
        .expect("phi4-14b stop");
    assert_eq!(stop4.dialect, PhiChatDialect::Phi4ImSep);
    assert_eq!(stop4.im_sep_token_id, Some(100_266));
    assert_eq!(stop4.im_end_token_id, Some(100_265));
    let crit4 = stop4.stop_criteria(32, 256);
    assert_eq!(
        crit4.evaluate(100_266, 1, 2),
        Some(StopReason::EosToken(100_266))
    );
    assert_eq!(
        crit4.evaluate(100_265, 1, 2),
        Some(StopReason::EosToken(100_265))
    );
}
