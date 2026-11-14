//! Native integration parity tests exercising full 256-dim super-block formats (`Q4_0`,
//! `Q5_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `F16`, `BF16`, `F32`) in end-to-end
//! `Phi3Recipe` forward passes without stalling Miri's `--lib` gate.

use jero_arch_phi3::{Phi3Recipe, RopeScalingConfig};
use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

struct ParityWeights<'a> {
    tensors: Vec<(String, TensorView<'a>)>,
}

impl ModelWeights for ParityWeights<'_> {
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
                reason: "tensor not found in parity weights".to_owned(),
            })
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.iter().map(|(_, v)| v.byte_len()).sum()
    }
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

fn encode_f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[expect(
    clippy::too_many_lines,
    reason = "independent 9-quant GGML reference decoder"
)]
fn ref_dequant_tensor(quant: QuantType, raw: &[u8], cols: usize, rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(cols * rows);
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
    assert_eq!(out.len(), cols * rows);
    out
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end 9-quant reference parity check"
)]
fn test_full_mixed_quant_forward_cosine_and_top1_parity() {
    const HIDDEN: usize = 256;
    const FFN: usize = 256;
    const VOCAB: usize = 4;
    const H_Q: usize = 8;
    const H_KV: usize = 2;
    const HEAD_DIM: usize = 32;
    const TOTAL_Q: usize = H_Q * HEAD_DIM;
    const TOTAL_KV: usize = H_KV * HEAD_DIM;

    let recipe = Phi3Recipe {
        architecture: "phi3",
        num_layers: 1,
        hidden_dim: HIDDEN,
        ffn_dim: FFN,
        n_heads_q: H_Q,
        n_heads_kv: H_KV,
        head_dim: HEAD_DIM,
        rotary_dim: Some(16),
        vocab_size: VOCAB,
        rms_eps: 1.0e-5,
        rope_theta: 10_000.0,
        rope_attn_factor: 1.1,
        original_max_position_embeddings: 8,
        sliding_window: None,
        rope_scaling: RopeScalingConfig::None,
    };

    let mut emb_raw = Vec::new();
    for r in 0..VOCAB {
        for b in 0..(HIDDEN / 32) {
            let mut blk = [0_u8; 18];
            blk[0..2].copy_from_slice(&f32_to_f16_bits(0.1).to_le_bytes());
            for (j, byte) in blk[2..18].iter_mut().enumerate() {
                let q0 = u8::try_from((r * 3 + b + j) % 15).unwrap_or(0);
                let q1 = u8::try_from((r * 5 + b + j + 2) % 15).unwrap_or(0);
                *byte = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
            emb_raw.extend_from_slice(&blk);
        }
    }

    let mut o_raw = Vec::new();
    for r in 0..HIDDEN {
        for b in 0..(TOTAL_Q / 32) {
            let mut blk = [0_u8; 22];
            blk[0..2].copy_from_slice(&f32_to_f16_bits(0.08).to_le_bytes());
            blk[2..6].copy_from_slice(&(0xA5A5_5A5A_u32).to_le_bytes());
            for (j, byte) in blk[6..22].iter_mut().enumerate() {
                let q0 = u8::try_from((r + b + j) % 15).unwrap_or(0);
                let q1 = u8::try_from((r * 3 + j) % 15).unwrap_or(0);
                *byte = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
            o_raw.extend_from_slice(&blk);
        }
    }

    let q5k_rows = |rows: usize, seed: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * 176);
        for r in 0..rows {
            let mut blk = vec![0_u8; 176];
            blk[0..2].copy_from_slice(&f32_to_f16_bits(0.15).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
            for (i, b) in blk[4..16].iter_mut().enumerate() {
                *b = u8::try_from((r + i + seed) % 5 + 1).unwrap_or(1);
            }
            for (i, b) in blk[16..48].iter_mut().enumerate() {
                *b = u8::try_from((r * 3 + i) & 0xFF).unwrap_or(0);
            }
            for (i, b) in blk[48..176].iter_mut().enumerate() {
                *b = u8::try_from(((r + i + seed) % 9) | (((r * 2 + i) % 9) << 4)).unwrap_or(0);
            }
            out.extend_from_slice(&blk);
        }
        out
    };
    let q4km_rows = |rows: usize, seed: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * 144);
        for r in 0..rows {
            let mut blk = vec![0_u8; 144];
            blk[0..2].copy_from_slice(&f32_to_f16_bits(0.2).to_le_bytes());
            blk[2..4].copy_from_slice(&f32_to_f16_bits(0.03).to_le_bytes());
            for (i, b) in blk[4..16].iter_mut().enumerate() {
                *b = u8::try_from((r + i + seed) % 4 + 1).unwrap_or(1);
            }
            for (i, b) in blk[16..144].iter_mut().enumerate() {
                *b = u8::try_from(((r + i + seed) % 7) | (((r * 5 + i) % 7) << 4)).unwrap_or(0);
            }
            out.extend_from_slice(&blk);
        }
        out
    };
    let q6k_rows = |rows: usize, seed: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * 210);
        for r in 0..rows {
            let mut blk = vec![0_u8; 210];
            for (i, b) in blk[0..128].iter_mut().enumerate() {
                *b = u8::try_from((r + i * 3 + seed) & 0xFF).unwrap_or(0);
            }
            for (i, b) in blk[128..192].iter_mut().enumerate() {
                *b = u8::try_from((r + i * 5) & 0xFF).unwrap_or(0);
            }
            for (i, b) in blk[192..208].iter_mut().enumerate() {
                let signed = i8::try_from((r + i) % 5).unwrap_or(0) - 2;
                *b = signed.to_ne_bytes()[0];
            }
            blk[208..210].copy_from_slice(&f32_to_f16_bits(0.05).to_le_bytes());
            out.extend_from_slice(&blk);
        }
        out
    };

    let mut q8_out = Vec::new();
    for r in 0..VOCAB {
        for b in 0..(HIDDEN / 32) {
            q8_out.extend_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
            for j in 0..32 {
                let signed = i8::try_from((r * 7 + b + j) % 19).unwrap_or(0) - 9;
                q8_out.push(signed.to_ne_bytes()[0]);
            }
        }
    }

    let mut bf16_norm = Vec::with_capacity(HIDDEN * 2);
    let mut f16_norm = Vec::with_capacity(HIDDEN * 2);
    for _ in 0..HIDDEN {
        bf16_norm.extend_from_slice(
            &u16::try_from(1.0_f32.to_bits() >> 16)
                .unwrap_or(0)
                .to_le_bytes(),
        );
        f16_norm.extend_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    }
    let f32_norm = encode_f32_bytes(&vec![1.0_f32; HIDDEN]);

    let mixed_specs = vec![
        (
            "token_embd.weight".to_owned(),
            QuantType::Q4_0,
            vec![HIDDEN, VOCAB],
            emb_raw,
        ),
        (
            "blk.0.attn_norm.weight".to_owned(),
            QuantType::BF16,
            vec![HIDDEN],
            bf16_norm,
        ),
        (
            "blk.0.attn_qkv.weight".to_owned(),
            QuantType::Q5_K,
            vec![HIDDEN, TOTAL_Q + 2 * TOTAL_KV],
            q5k_rows(TOTAL_Q + 2 * TOTAL_KV, 7),
        ),
        (
            "blk.0.attn_output.weight".to_owned(),
            QuantType::Q5_0,
            vec![TOTAL_Q, HIDDEN],
            o_raw,
        ),
        (
            "blk.0.ffn_norm.weight".to_owned(),
            QuantType::F16,
            vec![HIDDEN],
            f16_norm,
        ),
        (
            "blk.0.ffn_up.weight".to_owned(),
            QuantType::Q4_K_M,
            vec![HIDDEN, 2 * FFN],
            q4km_rows(2 * FFN, 19),
        ),
        (
            "blk.0.ffn_down.weight".to_owned(),
            QuantType::Q6_K,
            vec![FFN, HIDDEN],
            q6k_rows(HIDDEN, 23),
        ),
        (
            "output_norm.weight".to_owned(),
            QuantType::F32,
            vec![HIDDEN],
            f32_norm,
        ),
        (
            "output.weight".to_owned(),
            QuantType::Q8_0,
            vec![HIDDEN, VOCAB],
            q8_out,
        ),
    ];

    let f32_specs: Vec<(String, QuantType, Vec<usize>, Vec<u8>)> = mixed_specs
        .iter()
        .map(|(name, qt, shape, raw)| {
            let (cols, rows) = if shape.len() == 1 {
                (shape[0], 1)
            } else {
                (shape[0], shape[1])
            };
            let decoded = ref_dequant_tensor(*qt, raw, cols, rows);
            (
                name.clone(),
                QuantType::F32,
                shape.clone(),
                encode_f32_bytes(&decoded),
            )
        })
        .collect();

    let w_mixed = ParityWeights {
        tensors: mixed_specs
            .iter()
            .map(|(n, qt, s, r)| (n.clone(), TensorView::from_bytes(n, *qt, s, 0, r).unwrap()))
            .collect(),
    };
    let w_f32 = ParityWeights {
        tensors: f32_specs
            .iter()
            .map(|(n, qt, s, r)| (n.clone(), TensorView::from_bytes(n, *qt, s, 0, r).unwrap()))
            .collect(),
    };

    let mut logits_m = [0.0_f32; VOCAB];
    let mut kv_m = vec![0.0_f32; 8 * TOTAL_KV * 2];
    let mut ctx_m = ExecutionContext::new(&[1, 2], 8, &mut logits_m, &mut kv_m).expect("ctx m");
    recipe
        .prefill_step(&mut ctx_m, &w_mixed)
        .expect("mixed prefill");

    let mut logits_f = [0.0_f32; VOCAB];
    let mut kv_f = vec![0.0_f32; 8 * TOTAL_KV * 2];
    let mut ctx_f = ExecutionContext::new(&[1, 2], 8, &mut logits_f, &mut kv_f).expect("ctx f");
    recipe
        .prefill_step(&mut ctx_f, &w_f32)
        .expect("f32 prefill");

    let top_m = logits_m
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    let top_f = logits_f
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    assert_eq!(
        top_m, top_f,
        "greedy top-1 drifted: {logits_m:?} vs {logits_f:?}"
    );

    let dot: f32 = logits_m.iter().zip(&logits_f).map(|(&a, &b)| a * b).sum();
    let nm: f32 = logits_m.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let nf: f32 = logits_f.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let cos = dot / (nm * nf);
    assert!(cos > 0.9999, "cosine similarity too low: {cos}");
}
