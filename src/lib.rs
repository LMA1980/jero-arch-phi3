//! `jero-arch-phi3`: Microsoft Phi architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`jero_core::ArchRecipe`] for Microsoft Phi-3 and Phi-4 models (`phi3`, `phi4`),
//! supporting mixed-quantization GGUF execution, sliced fused `QKV`/`MLP` tensors,
//! `0..=pos` [`jero_core::KvCacheView`] causal GQA, tied/untied LM heads, and `LongRoPE`
//! context scaling.

pub(crate) mod quant;
pub mod recipe;
pub mod repack;

pub use recipe::{
    PHI4_14B_VOCAB_SIZE, PHI4_MINI_VOCAB_SIZE, PHI35_PADDED_VOCAB_SIZE, Phi3Recipe, PhiChatDialect,
    PhiStopMetadata, RopeScalingConfig,
};
pub use repack::{
    Phi3LayerDescriptor, Phi3LayerRepack, Phi3ModelRepack, Phi3RepackCache, Phi3SplitMatrix,
    REPACK_TILE_ALIGNMENT_BYTES, REPACK_TILE_ROWS,
};

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::gguf::{
        GgufArray, GgufFile, GgufValue, test_helpers::create_synthetic_gguf_v3_with_kvs,
    };
    use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

    struct TestWeights<'a> {
        tensors: Vec<(String, TensorView<'a>)>,
    }

    impl ModelWeights for TestWeights<'_> {
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
                    reason: "tensor not found in test weights".to_owned(),
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
        let unbiased = exp - 127;
        let f16_exp = unbiased + 15;
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

    fn f16_bits_to_f32(bits: u16) -> f32 {
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

    fn f32_to_bf16_bits(val: f32) -> u16 {
        u16::try_from(val.to_bits() >> 16).unwrap_or(0)
    }

    fn encode_f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn encode_f16_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        out
    }

    fn encode_bf16_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        out
    }

    fn encode_q8_0_rows(rows: &[Vec<f32>]) -> Vec<u8> {
        let mut raw = Vec::new();
        for row in rows {
            assert_eq!(row.len() % 32, 0);
            for chunk in row.chunks_exact(32) {
                let max_abs = chunk.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
                let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                let d_bits = f32_to_f16_bits(d);
                let d_quant = f16_bits_to_f32(d_bits);
                raw.extend_from_slice(&d_bits.to_le_bytes());
                for &v in chunk {
                    let q = if d_quant > 0.0 {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "clamped -127..=127 fits cleanly in i8"
                        )]
                        let rounded = (v / d_quant).round().clamp(-127.0, 127.0) as i8;
                        rounded
                    } else {
                        0_i8
                    };
                    raw.push(q.to_ne_bytes()[0]);
                }
            }
        }
        raw
    }

    fn encode_q4_0_block(vals: &[f32; 32]) -> [u8; 18] {
        let mut blk = [0_u8; 18];
        let max_abs = vals.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        let d = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
        let d_bits = f32_to_f16_bits(d);
        let d_q = f16_bits_to_f32(d_bits);
        blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for j in 0..16 {
            let q0 = if d_q > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped 0..=15"
                )]
                {
                    ((vals[j] / d_q).round().clamp(-8.0, 7.0) + 8.0) as u8
                }
            } else {
                8
            };
            let q1 = if d_q > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped 0..=15"
                )]
                {
                    ((vals[j + 16] / d_q).round().clamp(-8.0, 7.0) + 8.0) as u8
                }
            } else {
                8
            };
            blk[2 + j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        blk
    }

    fn encode_q5_0_block(vals: &[f32; 32]) -> [u8; 22] {
        let mut blk = [0_u8; 22];
        let max_abs = vals.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        let d = if max_abs > 0.0 { max_abs / 15.0 } else { 0.0 };
        let d_bits = f32_to_f16_bits(d);
        let d_q = f16_bits_to_f32(d_bits);
        blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
        let mut qh = 0_u32;
        for j in 0..16 {
            let q0 = if d_q > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped 0..=31"
                )]
                {
                    ((vals[j] / d_q).round().clamp(-16.0, 15.0) + 16.0) as u8
                }
            } else {
                16
            };
            let q1 = if d_q > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped 0..=31"
                )]
                {
                    ((vals[j + 16] / d_q).round().clamp(-16.0, 15.0) + 16.0) as u8
                }
            } else {
                16
            };
            qh |= u32::from((q0 >> 4) & 1) << j;
            qh |= u32::from((q1 >> 4) & 1) << (j + 16);
            blk[6 + j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        blk[2..6].copy_from_slice(&qh.to_le_bytes());
        blk
    }

    #[test]
    fn test_phi3_recipe_basic() {
        let recipe = Phi3Recipe::new();
        assert_eq!(recipe.architecture_id(), "phi3");
        assert_eq!(recipe.recipe_version(), "1.0.0");
        assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);
        assert!(recipe.supports_quant(QuantType::Q4_0));
        assert!(recipe.supports_quant(QuantType::Q5_0));
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q5_K));
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(!recipe.supports_quant(QuantType::IQ4_NL));
        assert!(!recipe.supports_quant(QuantType::Q2_K));
        assert_eq!(recipe.original_max_position_embeddings, 4096);
    }

    #[test]
    fn test_phi4_recipe_basic() {
        let recipe = Phi3Recipe::new_phi4();
        assert_eq!(recipe.architecture_id(), "phi4");
        assert_eq!(recipe.recipe_version(), "1.0.0");
        assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);
        assert_eq!(recipe.num_layers, 40);
        assert_eq!(recipe.hidden_dim, 5120);
        assert_eq!(recipe.n_heads_q, 40);
        assert_eq!(recipe.n_heads_kv, 10);
        assert_eq!(recipe.original_max_position_embeddings, 16384);
    }

    #[test]
    fn test_longrope_context_switching() {
        let short_factor = vec![1.0_f32; 48];
        let long_factor = vec![2.0_f32; 48];
        let scaling = RopeScalingConfig::LongRope {
            short_factor,
            long_factor,
            original_max_position_embeddings: 4096,
        };

        let recipe = Phi3Recipe::new().with_rope_scaling(scaling);

        assert!(!recipe.is_long_context(4096));
        let short_factors = recipe.active_rope_factors(4096).expect("factors available");
        assert!((short_factors[0] - 1.0).abs() < 1e-6);

        assert!(recipe.is_long_context(4097));
        let long_factors = recipe.active_rope_factors(4097).expect("factors available");
        assert!((long_factors[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_quant_admission_and_validate_model() {
        // phi3.5:3.8b (Q4_0) and phi4-mini:3.8b (Q4_K_M + Q5_K attn_qkv + Q5_0)
        let bytes_ok = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[
                ("phi3.block_count", GgufValue::Uint32(1)),
                ("phi3.embedding_length", GgufValue::Uint32(32)),
                ("phi3.attention.head_count", GgufValue::Uint32(4)),
                ("phi3.attention.head_count_kv", GgufValue::Uint32(2)),
                ("phi3.feed_forward_length", GgufValue::Uint32(64)),
            ],
            &[
                ("token_embd.weight", QuantType::Q4_0, &[32, 16]),
                ("blk.0.attn_qkv.weight", QuantType::Q5_K, &[256, 256]),
                ("blk.0.attn_output.weight", QuantType::Q5_0, &[32, 32]),
                ("blk.0.ffn_up.weight", QuantType::Q4_K_M, &[256, 256]),
                ("blk.0.ffn_down.weight", QuantType::Q6_K, &[256, 256]),
                ("blk.0.attn_norm.weight", QuantType::F16, &[32]),
                ("blk.0.ffn_norm.weight", QuantType::BF16, &[32]),
                ("output_norm.weight", QuantType::F32, &[32]),
            ],
        );
        let gguf_ok = GgufFile::parse(&bytes_ok).expect("parse mixed gguf");
        let recipe_ok = Phi3Recipe::from_gguf(&gguf_ok).expect("hydrate recipe");
        let empty = TestWeights { tensors: vec![] };
        recipe_ok
            .validate_model(gguf_ok.metadata(), &empty)
            .expect("mixed Q4_0/Q5_K/Q5_0/Q4_K_M/Q6_K admitted");

        // Unsupported tensor IQ4_NL must fail closed with CapabilityMismatch naming the tensor
        let bytes_bad = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[
                ("phi3.block_count", GgufValue::Uint32(1)),
                ("phi3.embedding_length", GgufValue::Uint32(32)),
                ("phi3.attention.head_count", GgufValue::Uint32(4)),
                ("phi3.attention.head_count_kv", GgufValue::Uint32(2)),
            ],
            &[("blk.0.attn_qkv.weight", QuantType::IQ4_NL, &[32, 64])],
        );
        let gguf_bad = GgufFile::parse(&bytes_bad).expect("parse bad gguf");
        let err = recipe_ok
            .validate_model(gguf_bad.metadata(), &empty)
            .expect_err("IQ4_NL must be rejected");
        match err {
            K3Error::CapabilityMismatch { reason, .. } => {
                assert!(
                    reason.contains("blk.0.attn_qkv.weight"),
                    "unexpected error: {reason}"
                );
            }
            other => panic!("expected CapabilityMismatch, got {other:?}"),
        }

        // Invalid GQA head divisibility must fail closed
        let mut bad_gqa = recipe_ok;
        bad_gqa.n_heads_q = 32;
        bad_gqa.n_heads_kv = 6;
        assert!(bad_gqa.validate_model(gguf_ok.metadata(), &empty).is_err());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "validates 3 full GGUF presets + failure"
    )]
    fn test_from_gguf_hydrates_phi35_phi4mini_phi4_and_longrope() {
        // 1. Phi-3.5-mini (3072 hidden, 32:32 MHA, 32064 vocab, LongRoPE Float32 + attn_factor)
        let short_35 = vec![1.05_f32; 48];
        let long_35 = vec![2.25_f32; 48];
        let b35 = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[
                ("phi3.block_count", GgufValue::Uint32(32)),
                ("phi3.embedding_length", GgufValue::Uint32(3072)),
                ("phi3.feed_forward_length", GgufValue::Uint32(8192)),
                ("phi3.attention.head_count", GgufValue::Uint32(32)),
                ("phi3.attention.head_count_kv", GgufValue::Uint32(32)),
                ("phi3.attention.key_length", GgufValue::Uint32(96)),
                ("phi3.context_length", GgufValue::Uint32(131_072)),
                ("phi3.vocab_size", GgufValue::Uint32(32_064)),
                ("phi3.rope.freq_base", GgufValue::Float32(10_000.0)),
                ("phi3.rope.scaling.type", GgufValue::String("longrope")),
                (
                    "phi3.rope.scaling.original_context_length",
                    GgufValue::Uint32(4096),
                ),
                ("phi3.rope.scaling.attn_factor", GgufValue::Float32(1.19)),
                (
                    "phi3.rope.scaling.short_factor",
                    GgufValue::Array(GgufArray::Float32(short_35)),
                ),
                (
                    "phi3.rope.scaling.long_factor",
                    GgufValue::Array(GgufArray::Float32(long_35)),
                ),
            ],
            &[],
        );
        let g35 = GgufFile::parse(&b35).expect("parse phi3.5");
        let r35 = Phi3Recipe::from_gguf(&g35).expect("hydrate phi3.5");
        assert_eq!(r35.architecture_id(), "phi3");
        assert_eq!(r35.hidden_dim, 3072);
        assert_eq!(r35.ffn_dim, 8192);
        assert_eq!(r35.n_heads_q, 32);
        assert_eq!(r35.n_heads_kv, 32);
        assert_eq!(r35.head_dim, 96);
        assert_eq!(r35.vocab_size, 32_064);
        assert!((r35.rope_attn_factor - 1.19).abs() < 1e-6);
        assert_eq!(r35.original_max_position_embeddings, 4096);
        assert!(matches!(
            r35.rope_scaling,
            RopeScalingConfig::LongRope { ref short_factor, ref long_factor, original_max_position_embeddings: 4096 }
            if short_factor.len() == 48 && long_factor.len() == 48
        ));

        // 2. Phi-4-mini (3072 hidden, 32:8 GQA, 200064 vocab, partial RoPE 64/96, Float64 factors)
        let mini_short = vec![1.0_f64; 32];
        let mini_long = vec![3.5_f64; 32];
        let b4m = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[
                ("phi3.block_count", GgufValue::Uint32(32)),
                ("phi3.embedding_length", GgufValue::Uint32(3072)),
                ("phi3.feed_forward_length", GgufValue::Uint32(8192)),
                ("phi3.attention.head_count", GgufValue::Uint32(32)),
                ("phi3.attention.head_count_kv", GgufValue::Uint32(8)),
                ("phi3.attention.key_length", GgufValue::Uint32(96)),
                ("phi3.rope.dimension_count", GgufValue::Uint32(64)),
                ("phi3.vocab_size", GgufValue::Uint32(200_064)),
                (
                    "phi3.rope.scaling.original_context_length",
                    GgufValue::Uint32(4096),
                ),
                ("phi3.rope.scaling.attn_factor", GgufValue::Float64(1.15)),
                (
                    "phi3.rope.scaling.short_factor",
                    GgufValue::Array(GgufArray::Float64(mini_short)),
                ),
                (
                    "phi3.rope.scaling.long_factor",
                    GgufValue::Array(GgufArray::Float64(mini_long)),
                ),
            ],
            &[],
        );
        let g4m = GgufFile::parse(&b4m).expect("parse phi4-mini");
        let r4m = Phi3Recipe::from_gguf(&g4m).expect("hydrate phi4-mini");
        assert_eq!(r4m.n_heads_q, 32);
        assert_eq!(r4m.n_heads_kv, 8);
        assert_eq!(r4m.vocab_size, 200_064);
        assert_eq!(r4m.rotary_dim, Some(64));
        assert_eq!(r4m.effective_rotary_dim(), 64);
        assert!((r4m.rope_attn_factor - 1.15).abs() < 1e-6);

        // 3. Phi-4 14B (phi4, 5120 hidden, 40:10 GQA, 17920 ffn, 100352 vocab, rope_theta 250k)
        let b4 = create_synthetic_gguf_v3_with_kvs(
            "phi4",
            32,
            &[
                ("phi4.block_count", GgufValue::Uint32(40)),
                ("phi4.embedding_length", GgufValue::Uint32(5120)),
                ("phi4.feed_forward_length", GgufValue::Uint32(17_920)),
                ("phi4.attention.head_count", GgufValue::Uint32(40)),
                ("phi4.attention.head_count_kv", GgufValue::Uint32(10)),
                ("phi4.attention.key_length", GgufValue::Uint32(128)),
                ("phi4.vocab_size", GgufValue::Uint32(100_352)),
                ("phi4.rope.freq_base", GgufValue::Float32(250_000.0)),
                ("phi4.context_length", GgufValue::Uint32(16_384)),
            ],
            &[],
        );
        let g4 = GgufFile::parse(&b4).expect("parse phi4-14b");
        let r4 = Phi3Recipe::from_gguf(&g4).expect("hydrate phi4-14b");
        assert_eq!(r4.architecture_id(), "phi4");
        assert_eq!(r4.hidden_dim, 5120);
        assert_eq!(r4.ffn_dim, 17_920);
        assert_eq!(r4.n_heads_q, 40);
        assert_eq!(r4.n_heads_kv, 10);
        assert_eq!(r4.head_dim, 128);
        assert_eq!(r4.vocab_size, 100_352);
        assert!((r4.rope_theta - 250_000.0).abs() < 1e-3);

        // Wrong LongRoPE factor length must fail closed
        let bad_lr = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[
                ("phi3.block_count", GgufValue::Uint32(1)),
                ("phi3.embedding_length", GgufValue::Uint32(32)),
                ("phi3.attention.head_count", GgufValue::Uint32(4)),
                ("phi3.attention.head_count_kv", GgufValue::Uint32(4)),
                (
                    "phi3.rope.scaling.short_factor",
                    GgufValue::Array(GgufArray::Float32(vec![1.0; 99])),
                ),
                (
                    "phi3.rope.scaling.long_factor",
                    GgufValue::Array(GgufArray::Float32(vec![2.0; 99])),
                ),
            ],
            &[],
        );
        let g_bad = GgufFile::parse(&bad_lr).expect("parse bad lr");
        assert!(Phi3Recipe::from_gguf(&g_bad).is_err());
    }

    #[test]
    fn test_all_admitted_quants_embedding_and_projection_parity() {
        const COLS: usize = 256;
        let mut x = vec![0.0_f32; COLS];
        for (i, slot) in x.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small loop index fits in f32")]
            {
                *slot = ((i % 13) as f32 - 6.0) * 0.125;
            }
        }

        // Test unquantized F32/F16/BF16 & 32-elem block Q4_0/Q5_0/Q8_0 & 256-elem Q4_K_M/Q5_K/Q6_K
        let row_f32: Vec<f32> = (0..COLS)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small loop index fits in f32")]
                let val = ((i % 11) as f32 - 5.0) * 0.25;
                val
            })
            .collect();

        let mut q4_0_bytes = Vec::new();
        let mut q5_0_bytes = Vec::new();
        for chunk in row_f32.chunks_exact(32) {
            let mut arr = [0.0_f32; 32];
            arr.copy_from_slice(chunk);
            q4_0_bytes.extend_from_slice(&encode_q4_0_block(&arr));
            q5_0_bytes.extend_from_slice(&encode_q5_0_block(&arr));
        }

        // Valid single-superblock payloads for Q4_K_M (144B), Q5_K (176B), Q6_K (210B)
        let mut q4k = vec![0_u8; 144];
        q4k[0..2].copy_from_slice(&f32_to_f16_bits(0.5).to_le_bytes()); // d = 0.5
        q4k[2..4].copy_from_slice(&f32_to_f16_bits(0.1).to_le_bytes()); // dmin = 0.1
        for b in &mut q4k[4..16] {
            *b = 0x05;
        }
        for (i, b) in q4k[16..144].iter_mut().enumerate() {
            *b = u8::try_from((i % 7) | (((i + 3) % 7) << 4)).unwrap_or(0);
        }

        let mut q5k = vec![0_u8; 176];
        q5k[0..2].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes());
        q5k[2..4].copy_from_slice(&f32_to_f16_bits(0.05).to_le_bytes());
        for b in &mut q5k[4..16] {
            *b = 0x03;
        }
        for b in &mut q5k[16..48] {
            *b = 0xAA;
        }
        for (i, b) in q5k[48..176].iter_mut().enumerate() {
            *b = u8::try_from((i % 9) | (((i + 1) % 9) << 4)).unwrap_or(0);
        }

        let mut q6k = vec![0_u8; 210];
        for (i, b) in q6k[0..128].iter_mut().enumerate() {
            *b = u8::try_from((i * 3) & 0xFF).unwrap_or(0);
        }
        for (i, b) in q6k[128..192].iter_mut().enumerate() {
            *b = u8::try_from((i * 5) & 0xFF).unwrap_or(0);
        }
        for (i, b) in q6k[192..208].iter_mut().enumerate() {
            let signed = i8::try_from(i % 7).unwrap_or(0) - 3;
            *b = signed.to_ne_bytes()[0];
        }
        q6k[208..210].copy_from_slice(&f32_to_f16_bits(0.125).to_le_bytes());

        let cases: [(&str, QuantType, Vec<u8>); 9] = [
            ("F32", QuantType::F32, encode_f32_bytes(&row_f32)),
            ("F16", QuantType::F16, encode_f16_bytes(&row_f32)),
            ("BF16", QuantType::BF16, encode_bf16_bytes(&row_f32)),
            ("Q4_0", QuantType::Q4_0, q4_0_bytes),
            ("Q5_0", QuantType::Q5_0, q5_0_bytes),
            (
                "Q8_0",
                QuantType::Q8_0,
                encode_q8_0_rows(std::slice::from_ref(&row_f32)),
            ),
            ("Q4_K_M", QuantType::Q4_K_M, q4k),
            ("Q5_K", QuantType::Q5_K, q5k),
            ("Q6_K", QuantType::Q6_K, q6k),
        ];

        let shape = [COLS, 1];
        for (label, quant, bytes) in cases {
            let tv = TensorView::from_bytes("w", quant, &shape, 0, &bytes).expect("view");
            let mut decoded = vec![0.0_f32; COLS];
            quant::dequantize_row_to_f32(&tv, 0, COLS, &mut decoded)
                .unwrap_or_else(|e| panic!("{label} dequant failed: {e:?}"));

            let ref_dot: f32 = decoded.iter().zip(&x).map(|(&a, &b)| a * b).sum();
            let mut proj = [0.0_f32; 1];
            quant::project_matvec(&mut proj, &tv, &x)
                .unwrap_or_else(|e| panic!("{label} project failed: {e:?}"));
            assert!(
                (proj[0] - ref_dot).abs() < 1e-4,
                "{label} dot parity mismatch: proj={} ref={ref_dot}",
                proj[0]
            );
            assert!(
                decoded.iter().any(|&v| v.abs() > 1e-6),
                "{label} decoded row unexpectedly all zeros"
            );
        }
    }

    fn build_1layer_recipe(
        hidden: usize,
        ffn: usize,
        h_q: usize,
        h_kv: usize,
        vocab: usize,
    ) -> Phi3Recipe {
        let mut r = Phi3Recipe::new();
        r.num_layers = 1;
        r.hidden_dim = hidden;
        r.ffn_dim = ffn;
        r.n_heads_q = h_q;
        r.n_heads_kv = h_kv;
        r.head_dim = hidden / h_q;
        r.vocab_size = vocab;
        r.rope_theta = 10_000.0;
        r.rms_eps = 1.0e-5;
        r.original_max_position_embeddings = 64;
        r
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "test fixture reference builder"
    )]
    fn build_full_weight_buffers(
        hidden: usize,
        ffn: usize,
        total_q: usize,
        total_kv: usize,
        vocab: usize,
        norm_q_type: QuantType,
        proj_q_type: QuantType,
        untied_out: bool,
    ) -> Vec<(String, QuantType, Vec<usize>, Vec<u8>)> {
        let mk_matrix = |rows: usize, cols: usize, seed: usize| -> Vec<Vec<f32>> {
            (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|c| {
                            #[expect(clippy::cast_precision_loss, reason = "small test dim")]
                            let val = (((r * 7 + c * 3 + seed) % 15) as f32 - 7.0) * 0.05;
                            val
                        })
                        .collect()
                })
                .collect()
        };
        let encode_mat = |qt: QuantType, rows: &[Vec<f32>]| -> Vec<u8> {
            match qt {
                QuantType::F32 => {
                    let mut b = Vec::new();
                    for r in rows {
                        b.extend_from_slice(&encode_f32_bytes(r));
                    }
                    b
                }
                _ => encode_q8_0_rows(rows),
            }
        };
        let encode_norm = |qt: QuantType, vals: &[f32]| -> Vec<u8> {
            match qt {
                QuantType::F16 => encode_f16_bytes(vals),
                QuantType::BF16 => encode_bf16_bytes(vals),
                _ => encode_f32_bytes(vals),
            }
        };

        let embd_rows = mk_matrix(vocab, hidden, 1);
        let qkv_rows = mk_matrix(total_q + 2 * total_kv, hidden, 11);
        let o_rows = mk_matrix(hidden, total_q, 23);
        let up_rows = mk_matrix(2 * ffn, hidden, 37);
        let down_rows = mk_matrix(hidden, ffn, 51);
        let out_rows = mk_matrix(vocab, hidden, 71);
        let norm_vals: Vec<f32> = (0..hidden)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test dim")]
                let val = 0.8 + ((i % 5) as f32) * 0.1;
                val
            })
            .collect();

        let mut list = vec![
            (
                "token_embd.weight".to_owned(),
                proj_q_type,
                vec![hidden, vocab],
                encode_mat(proj_q_type, &embd_rows),
            ),
            (
                "blk.0.attn_norm.weight".to_owned(),
                norm_q_type,
                vec![hidden],
                encode_norm(norm_q_type, &norm_vals),
            ),
            (
                "blk.0.attn_qkv.weight".to_owned(),
                proj_q_type,
                vec![hidden, total_q + 2 * total_kv],
                encode_mat(proj_q_type, &qkv_rows),
            ),
            (
                "blk.0.attn_output.weight".to_owned(),
                QuantType::F32,
                vec![total_q, hidden],
                encode_mat(QuantType::F32, &o_rows),
            ),
            (
                "blk.0.ffn_norm.weight".to_owned(),
                norm_q_type,
                vec![hidden],
                encode_norm(norm_q_type, &norm_vals),
            ),
            (
                "blk.0.ffn_up.weight".to_owned(),
                proj_q_type,
                vec![hidden, 2 * ffn],
                encode_mat(proj_q_type, &up_rows),
            ),
            (
                "blk.0.ffn_down.weight".to_owned(),
                QuantType::F32,
                vec![ffn, hidden],
                encode_mat(QuantType::F32, &down_rows),
            ),
            (
                "output_norm.weight".to_owned(),
                norm_q_type,
                vec![hidden],
                encode_norm(norm_q_type, &norm_vals),
            ),
        ];
        if untied_out {
            list.push((
                "output.weight".to_owned(),
                QuantType::F32,
                vec![hidden, vocab],
                encode_mat(QuantType::F32, &out_rows),
            ));
        }
        list
    }

    fn weights_from_buffers(specs: &[(String, QuantType, Vec<usize>, Vec<u8>)]) -> TestWeights<'_> {
        let mut tensors = Vec::with_capacity(specs.len());
        for (name, quant, shape, bytes) in specs {
            let tv = TensorView::from_bytes(name, *quant, shape, 0, bytes).expect("tensor view");
            tensors.push((name.clone(), tv));
        }
        TestWeights { tensors }
    }

    #[test]
    fn test_phi3_prefill_and_decode_step_with_fused_slicing() {
        let recipe = build_1layer_recipe(8, 16, 4, 2, 8);
        let specs = build_full_weight_buffers(8, 16, 8, 4, 8, QuantType::F32, QuantType::F32, true);
        let weights = weights_from_buffers(&specs);

        let prompt = [1_u32, 3, 5];
        let mut logits_batched = [0.0_f32; 8];
        let mut kv_batched = vec![0.0_f32; 16 * 4 * 2];
        let mut ctx_b = ExecutionContext::new(&prompt, 16, &mut logits_batched, &mut kv_batched)
            .expect("create batched ctx");
        recipe
            .prefill_step(&mut ctx_b, &weights)
            .expect("batched prefill succeeds");
        assert_eq!(ctx_b.position(), 3);
        assert!(logits_batched.iter().all(|v| v.is_finite()));

        // Batched prefill([1,3,5]) must match sequential prefill([1]) -> decode(3) -> decode(5)
        let mut logits_seq = [0.0_f32; 8];
        let mut kv_seq = vec![0.0_f32; 16 * 4 * 2];
        let mut ctx_s =
            ExecutionContext::new(&[1], 16, &mut logits_seq, &mut kv_seq).expect("create seq ctx");
        recipe.prefill_step(&mut ctx_s, &weights).expect("p0");
        ctx_s.prepare_decode(3).expect("prep d1");
        recipe.decode_step(&mut ctx_s, &weights).expect("d1");
        ctx_s.prepare_decode(5).expect("prep d2");
        recipe.decode_step(&mut ctx_s, &weights).expect("d2");
        assert_eq!(ctx_s.position(), 3);

        for (i, (&b, &s)) in logits_batched.iter().zip(&logits_seq).enumerate() {
            assert!(
                (b - s).abs() < 1e-5,
                "logit[{i}] batched {b} != sequential {s}"
            );
        }
    }

    #[test]
    fn test_synthetic_empty_weights_prefill_and_decode() {
        let recipe = build_1layer_recipe(8, 16, 4, 2, 16);
        let empty = TestWeights { tensors: vec![] };
        let mut logits = [0.0_f32; 16];
        let mut kv = vec![0.0_f32; 16 * 4 * 2];
        let mut ctx = ExecutionContext::new(&[2, 4, 6], 16, &mut logits, &mut kv).expect("ctx");
        recipe.prefill_step(&mut ctx, &empty).expect("prefill");
        assert_eq!(ctx.position(), 3);
        ctx.prepare_decode(5).expect("prep decode");
        recipe.decode_step(&mut ctx, &empty).expect("decode");
        assert_eq!(ctx.position(), 4);
        assert!(logits.iter().any(|&v| v.abs() > 1e-4));
    }

    #[test]
    fn test_multi_dtype_rmsnorm_and_tied_vs_untied_lm_head() {
        let recipe = build_1layer_recipe(8, 16, 4, 2, 6);
        for norm_qt in [QuantType::F32, QuantType::F16, QuantType::BF16] {
            let tied_specs =
                build_full_weight_buffers(8, 16, 8, 4, 6, norm_qt, QuantType::F32, false);
            let tied_weights = weights_from_buffers(&tied_specs);
            let mut logits = [0.0_f32; 6];
            let mut kv = vec![0.0_f32; 8 * 4 * 2];
            let mut ctx = ExecutionContext::new(&[1, 2], 8, &mut logits, &mut kv).expect("ctx");
            recipe
                .prefill_step(&mut ctx, &tied_weights)
                .expect("tied LM head succeeds");
            assert!(logits.iter().all(|v| v.is_finite()));
            assert!(logits.iter().any(|&v| v.abs() > 1e-6));
        }

        // Fail closed when partial loaded weights omit required token_embd.weight
        let partial_specs =
            build_full_weight_buffers(8, 16, 8, 4, 6, QuantType::F32, QuantType::F32, true)
                .into_iter()
                .filter(|(n, _, _, _)| n != "token_embd.weight")
                .collect::<Vec<_>>();
        let partial_weights = weights_from_buffers(&partial_specs);
        let mut logits = [0.0_f32; 6];
        let mut kv = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx = ExecutionContext::new(&[0], 8, &mut logits, &mut kv).expect("ctx");
        assert!(matches!(
            recipe.prefill_step(&mut ctx, &partial_weights),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_causal_gqa_kv_history_and_longrope_attn_factor() {
        let specs = build_full_weight_buffers(8, 16, 8, 4, 8, QuantType::F16, QuantType::F32, true);
        let weights = weights_from_buffers(&specs);

        let recipe = build_1layer_recipe(8, 16, 2, 1, 8)
            .with_rotary_dim(Some(4))
            .with_rope_scaling(RopeScalingConfig::LongRope {
                short_factor: vec![1.0, 1.2],
                long_factor: vec![2.0, 2.4],
                original_max_position_embeddings: 2,
            })
            .with_attn_factor(1.15);

        // Prompt [0, 4] vs [3, 4]: pos 1 evaluates token 4 under different causal prefixes 0 vs 3
        let mut logits_a = [0.0_f32; 8];
        let mut kv_a = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx_a = ExecutionContext::new(&[0, 4], 8, &mut logits_a, &mut kv_a).expect("ctx a");
        recipe.prefill_step(&mut ctx_a, &weights).expect("run a");

        let mut logits_b = [0.0_f32; 8];
        let mut kv_b = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx_b = ExecutionContext::new(&[3, 4], 8, &mut logits_b, &mut kv_b).expect("ctx b");
        recipe.prefill_step(&mut ctx_b, &weights).expect("run b");

        let max_diff = logits_a
            .iter()
            .zip(&logits_b)
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            max_diff > 1e-4,
            "pos=1 logits failed causal KV dependence check (max_diff={max_diff})"
        );

        // attn_factor scaling must alter output even at pos=0 vs attn_factor = 1.0
        let recipe_no_factor = recipe.clone().with_attn_factor(1.0);
        let mut logits_f1 = [0.0_f32; 8];
        let mut kv_f1 = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx_f1 = ExecutionContext::new(&[0, 4], 8, &mut logits_f1, &mut kv_f1).expect("f1");
        recipe_no_factor
            .prefill_step(&mut ctx_f1, &weights)
            .expect("run f1");

        let factor_diff = logits_a
            .iter()
            .zip(&logits_f1)
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            factor_diff > 1e-5,
            "rope_attn_factor had zero effect (diff={factor_diff})"
        );
    }

    #[test]
    fn test_full_ffn_dim_not_truncated() {
        // ffn_dim = 24 > hidden_dim = 8: perturb only rows 8..24 in gate and 32..48 in up
        let recipe = build_1layer_recipe(8, 24, 2, 1, 6);
        let specs_a =
            build_full_weight_buffers(8, 24, 8, 4, 6, QuantType::F32, QuantType::F32, true);
        let mut specs_b = specs_a.clone();
        // Zero out the upper FFN rows (>= hidden_dim = 8) of Gate [8..24] & Up [32..48]
        let up_entry = specs_b
            .iter_mut()
            .find(|(n, _, _, _)| n == "blk.0.ffn_up.weight")
            .expect("ffn_up present");
        let row_bytes = 8 * 4;
        for row_idx in (8..24).chain(32..48) {
            let start = row_idx * row_bytes;
            up_entry.3[start..start + row_bytes].fill(0);
        }

        let w_a = weights_from_buffers(&specs_a);
        let w_b = weights_from_buffers(&specs_b);

        let mut logits_a = [0.0_f32; 6];
        let mut kv_a = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx_a = ExecutionContext::new(&[2], 8, &mut logits_a, &mut kv_a).expect("ctx a");
        recipe.prefill_step(&mut ctx_a, &w_a).expect("run a");

        let mut logits_b = [0.0_f32; 6];
        let mut kv_b = vec![0.0_f32; 8 * 4 * 2];
        let mut ctx_b = ExecutionContext::new(&[2], 8, &mut logits_b, &mut kv_b).expect("ctx b");
        recipe.prefill_step(&mut ctx_b, &w_b).expect("run b");

        let diff = logits_a
            .iter()
            .zip(&logits_b)
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            diff > 1e-4,
            "rows beyond hidden_dim (8..24) in FFN were truncated (diff={diff})"
        );
    }

    #[test]
    fn test_mixed_quant_32d_forward_miri() {
        // Compact mixed-quant (Q4_0 + Q5_0 + Q8_0 + BF16 + F16 + F32) GQA forward pass (<1s under Miri)
        const HIDDEN: usize = 32;
        const FFN: usize = 32;
        const VOCAB: usize = 4;
        const H_Q: usize = 4;
        const H_KV: usize = 2;
        const HEAD_DIM: usize = 8;
        const TOTAL_Q: usize = H_Q * HEAD_DIM;
        const TOTAL_KV: usize = H_KV * HEAD_DIM;

        let recipe = build_1layer_recipe(HIDDEN, FFN, H_Q, H_KV, VOCAB).with_rotary_dim(Some(4));
        let norm_ones = vec![1.0_f32; HIDDEN];
        let mut emb_raw = Vec::new();
        for r in 0..VOCAB {
            let mut blk = [0.0_f32; 32];
            for (j, slot) in blk.iter_mut().enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "small test idx")]
                {
                    *slot = (((r * 5 + j) % 9) as f32 - 4.0) * 0.1;
                }
            }
            emb_raw.extend_from_slice(&encode_q4_0_block(&blk));
        }
        let mut o_raw = Vec::new();
        for r in 0..HIDDEN {
            let mut blk = [0.0_f32; 32];
            for (j, slot) in blk.iter_mut().enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "small test idx")]
                {
                    *slot = (((r * 3 + j) % 11) as f32 - 5.0) * 0.05;
                }
            }
            o_raw.extend_from_slice(&encode_q5_0_block(&blk));
        }

        let base_specs = build_full_weight_buffers(
            HIDDEN,
            FFN,
            TOTAL_Q,
            TOTAL_KV,
            VOCAB,
            QuantType::F32,
            QuantType::Q8_0,
            true,
        );
        let mut mixed_specs = Vec::with_capacity(base_specs.len());
        for (name, qt, shape, raw) in base_specs {
            match name.as_str() {
                "token_embd.weight" => {
                    mixed_specs.push((name, QuantType::Q4_0, shape, emb_raw.clone()));
                }
                "blk.0.attn_norm.weight" => {
                    mixed_specs.push((name, QuantType::BF16, shape, encode_bf16_bytes(&norm_ones)));
                }
                "blk.0.attn_output.weight" => {
                    mixed_specs.push((name, QuantType::Q5_0, shape, o_raw.clone()));
                }
                "blk.0.ffn_norm.weight" => {
                    mixed_specs.push((name, QuantType::F16, shape, encode_f16_bytes(&norm_ones)));
                }
                _ => mixed_specs.push((name, qt, shape, raw)),
            }
        }
        let weights = weights_from_buffers(&mixed_specs);
        let mut logits = [0.0_f32; VOCAB];
        let mut kv = vec![0.0_f32; 8 * TOTAL_KV * 2];
        let mut ctx = ExecutionContext::new(&[2], 8, &mut logits, &mut kv).expect("ctx");
        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("mixed forward");
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(logits.iter().any(|&v| v.abs() > 1e-6));
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

    fn pattern_f32(
        idx: usize,
        mul: usize,
        add: usize,
        modulus: usize,
        center: f32,
        scale: f32,
    ) -> f32 {
        let bucket = u8::try_from((idx * mul + add) % modulus).expect("modulus fits u8");
        (f32::from(bucket) - center) * scale
    }

    type SyntheticTensorSpec = (String, QuantType, Vec<usize>, Vec<u8>);

    fn build_gqa40_repack_specs(
        q_dim: usize,
        kv_dim: usize,
        ffn: usize,
        vocab: usize,
    ) -> Vec<SyntheticTensorSpec> {
        const HIDDEN: usize = 32;
        let mut specs = Vec::new();
        let embd_vals: Vec<f32> = (0..vocab * HIDDEN)
            .map(|i| pattern_f32(i, 7, 3, 19, 9.0, 0.05))
            .collect();
        specs.push((
            "token_embd.weight".to_owned(),
            QuantType::F32,
            vec![HIDDEN, vocab],
            encode_f32_bytes(&embd_vals),
        ));
        specs.push((
            "output_norm.weight".to_owned(),
            QuantType::BF16,
            vec![HIDDEN],
            encode_bf16_bytes(&[1.0_f32; HIDDEN]),
        ));
        specs.push((
            "blk.0.attn_norm.weight".to_owned(),
            QuantType::F16,
            vec![HIDDEN],
            encode_f16_bytes(&[1.0_f32; HIDDEN]),
        ));
        specs.push((
            "blk.0.ffn_norm.weight".to_owned(),
            QuantType::F32,
            vec![HIDDEN],
            encode_f32_bytes(&[1.0_f32; HIDDEN]),
        ));

        let qkv_rows = q_dim + 2 * kv_dim;
        let mut qkv_bytes = Vec::with_capacity(qkv_rows * 18);
        for r in 0..qkv_rows {
            let mut row = [0.0_f32; HIDDEN];
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = pattern_f32(r * 13 + c * 5, 1, 0, 23, 11.0, 0.04);
            }
            qkv_bytes.extend_from_slice(&encode_q4_0_block(&row));
        }
        specs.push((
            "blk.0.attn_qkv.weight".to_owned(),
            QuantType::Q4_0,
            vec![HIDDEN, qkv_rows],
            qkv_bytes,
        ));

        let o_flat: Vec<f32> = (0..HIDDEN)
            .flat_map(|r| (0..q_dim).map(move |c| pattern_f32(r * 11 + c * 3, 1, 0, 17, 8.0, 0.03)))
            .collect();
        specs.push((
            "blk.0.attn_output.weight".to_owned(),
            QuantType::F16,
            vec![q_dim, HIDDEN],
            encode_f16_bytes(&o_flat),
        ));

        let gate_up_rows: Vec<Vec<f32>> = (0..2 * ffn)
            .map(|r| {
                (0..HIDDEN)
                    .map(|c| pattern_f32(r * 17 + c * 7, 1, 0, 29, 14.0, 0.03))
                    .collect()
            })
            .collect();
        specs.push((
            "blk.0.ffn_up.weight".to_owned(),
            QuantType::Q8_0,
            vec![HIDDEN, 2 * ffn],
            encode_q8_0_rows(&gate_up_rows),
        ));

        let down_flat: Vec<f32> = (0..HIDDEN * ffn)
            .map(|i| pattern_f32(i, 9, 1, 15, 7.0, 0.035))
            .collect();
        specs.push((
            "blk.0.ffn_down.weight".to_owned(),
            QuantType::BF16,
            vec![ffn, HIDDEN],
            encode_bf16_bytes(&down_flat),
        ));
        specs
    }

    fn assert_split_row_parity(
        weights: &TestWeights,
        cache: &Phi3ModelRepack,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn: usize,
    ) {
        assert_eq!(cache.layers().len(), 1);
        let l0 = cache.layer(0).expect("layer 0");

        assert_eq!(l0.wq().rows(), q_dim);
        assert_eq!(l0.wq().n_tiles(), q_dim.div_ceil(REPACK_TILE_ROWS));
        assert_eq!(l0.wk().rows(), kv_dim);
        assert_eq!(l0.wk().n_tiles(), kv_dim.div_ceil(REPACK_TILE_ROWS));
        assert_eq!(l0.wv().rows(), kv_dim);
        assert_eq!(l0.wv().n_tiles(), kv_dim.div_ceil(REPACK_TILE_ROWS));
        assert_eq!(l0.wgate().rows(), ffn);
        assert_eq!(l0.wgate().n_tiles(), ffn.div_ceil(REPACK_TILE_ROWS));
        assert_eq!(l0.wup().rows(), ffn);
        assert_eq!(l0.wup().n_tiles(), ffn.div_ceil(REPACK_TILE_ROWS));
        for mat in [
            l0.wq(),
            l0.wk(),
            l0.wv(),
            l0.wo(),
            l0.wgate(),
            l0.wup(),
            l0.wdown(),
        ] {
            if mat.n_tiles() > 0 {
                assert!(mat.is_128_byte_aligned(), "{} not 128B aligned", mat.name());
            }
        }

        let qkv_view = weights.tensor("blk.0.attn_qkv.weight").expect("qkv view");
        let mut raw_row = vec![0.0_f32; hidden];
        let mut split_row = vec![0.0_f32; hidden];
        for r in [0, 7, 8, kv_dim - 1] {
            quant::dequantize_row_to_f32(&qkv_view, q_dim + r, hidden, &mut raw_row)
                .expect("raw k row");
            l0.wk().dequantize_row(r, &mut split_row).expect("wk row");
            let max_k = raw_row
                .iter()
                .zip(&split_row)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(max_k <= 1e-5, "wk row {r} err={max_k}");

            quant::dequantize_row_to_f32(&qkv_view, q_dim + kv_dim + r, hidden, &mut raw_row)
                .expect("raw v row");
            l0.wv().dequantize_row(r, &mut split_row).expect("wv row");
            let max_v = raw_row
                .iter()
                .zip(&split_row)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(max_v <= 1e-5, "wv row {r} err={max_v}");
        }

        let up_view = weights.tensor("blk.0.ffn_up.weight").expect("up view");
        for r in [0, 7, 8, ffn - 1] {
            quant::dequantize_row_to_f32(&up_view, ffn + r, hidden, &mut raw_row)
                .expect("raw up row");
            l0.wup().dequantize_row(r, &mut split_row).expect("wup row");
            let max_up = raw_row
                .iter()
                .zip(&split_row)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(max_up <= 1e-5, "wup row {r} err={max_up}");
        }

        assert_eq!(l0.attn_norm(), &vec![1.0_f32; hidden]);
        assert_eq!(l0.ffn_norm(), &vec![1.0_f32; hidden]);
        assert_eq!(cache.output_norm(), &vec![1.0_f32; hidden]);
    }

    #[test]
    fn test_presplit_fused_qkv_and_gate_up_zero_straddling_and_repack_forward_miri() {
        const HIDDEN: usize = 32;
        const HEADS_Q: usize = 40;
        const HEADS_KV: usize = 10;
        const HEAD_DIM: usize = 2;
        const Q_DIM: usize = HEADS_Q * HEAD_DIM;
        const KV_DIM: usize = HEADS_KV * HEAD_DIM;
        const FFN: usize = 12;
        const VOCAB: usize = 8;

        let mut recipe = Phi3Recipe::new();
        recipe.num_layers = 1;
        recipe.hidden_dim = HIDDEN;
        recipe.ffn_dim = FFN;
        recipe.n_heads_q = HEADS_Q;
        recipe.n_heads_kv = HEADS_KV;
        recipe.head_dim = HEAD_DIM;
        recipe.rotary_dim = Some(HEAD_DIM);
        recipe.vocab_size = VOCAB;

        let specs = build_gqa40_repack_specs(Q_DIM, KV_DIM, FFN, VOCAB);
        let weights = weights_from_buffers(&specs);
        let cache = recipe.build_repack_cache(&weights).expect("build repack");
        assert_split_row_parity(&weights, &cache, HIDDEN, Q_DIM, KV_DIM, FFN);

        let mut logits_uncached = [0.0_f32; VOCAB];
        let mut kv_uncached = vec![0.0_f32; 8 * KV_DIM * 2];
        let mut ctx_u =
            ExecutionContext::new(&[1, 3], 8, &mut logits_uncached, &mut kv_uncached).expect("u");
        recipe
            .prefill_step(&mut ctx_u, &weights)
            .expect("u prefill");
        ctx_u.prepare_decode(5).expect("u prep decode");
        recipe.decode_step(&mut ctx_u, &weights).expect("u decode");

        let mut logits_cached = [0.0_f32; VOCAB];
        let mut kv_cached = vec![0.0_f32; 8 * KV_DIM * 2];
        let mut ctx_c =
            ExecutionContext::new(&[1, 3], 8, &mut logits_cached, &mut kv_cached).expect("c");
        recipe
            .prefill_step_with_repack(&mut ctx_c, &weights, &cache)
            .expect("c prefill");
        ctx_c.prepare_decode(5).expect("c prep decode");
        recipe
            .decode_step_with_repack(&mut ctx_c, &weights, &cache)
            .expect("c decode");

        let cos = cosine_sim(&logits_cached, &logits_uncached);
        assert!(cos > 0.998, "repack decode cosine {cos} too low");
        assert_eq!(argmax_idx(&logits_cached), argmax_idx(&logits_uncached));

        let empty = TestWeights { tensors: vec![] };
        assert!(matches!(
            recipe.build_repack_cache(&empty),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_phi35_phi4_stop_metadata_and_dialect_handoff() {
        use jero_core::StopReason;

        let phi35_vocab_u32 = u32::try_from(PHI35_PADDED_VOCAB_SIZE).expect("fits u32");
        let phi35_tokens = vec![
            "<unk>",
            "<s>",
            "</s>",
            "<|user|>",
            "<|assistant|>",
            "hello",
            "<|end|>",
            "<|endoftext|>",
        ];
        let phi35_kvs = [
            ("phi3.vocab_size", GgufValue::Uint32(phi35_vocab_u32)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(phi35_tokens)),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(6)),
            (
                "tokenizer.chat_template",
                GgufValue::String(
                    "{% for m in messages %}<|user|>{{m.content}}<|end|>{% endfor %}",
                ),
            ),
        ];
        let phi35_bytes = create_synthetic_gguf_v3_with_kvs("phi3", 32, &phi35_kvs, &[]);
        let phi35_gguf = GgufFile::parse(&phi35_bytes).expect("parse phi35");
        let phi35_recipe = Phi3Recipe::from_gguf(&phi35_gguf).expect("hydrate phi35");
        assert_eq!(phi35_recipe.vocab_size, PHI35_PADDED_VOCAB_SIZE);
        let phi35_stop = phi35_recipe
            .extract_stop_metadata(&phi35_gguf)
            .expect("stop metadata phi35");
        assert_eq!(phi35_stop.dialect, PhiChatDialect::Phi3End);
        assert_eq!(phi35_stop.end_token_id, Some(6));
        assert!(phi35_stop.has_tied_lm_head);
        assert!(phi35_stop.is_stop_token(6));
        assert!(phi35_stop.is_stop_token(7));
        let crit35 = phi35_stop.stop_criteria(16, 128);
        assert_eq!(crit35.evaluate(5, 1, 2), None);
        assert_eq!(crit35.evaluate(6, 2, 3), Some(StopReason::EosToken(6)));

        let phi4_14b_u32 = u32::try_from(PHI4_14B_VOCAB_SIZE).expect("fits u32");
        let phi4_tokens = vec![
            "<unk>",
            "ok",
            "<|im_start|>",
            "<|im_sep|>",
            "<|im_end|>",
            "<|endoftext|>",
        ];
        let phi4_kvs = [
            ("phi4.vocab_size", GgufValue::Uint32(phi4_14b_u32)),
            ("phi4.embedding_length", GgufValue::Uint32(5120)),
            ("phi4.attention.head_count", GgufValue::Uint32(40)),
            ("phi4.attention.head_count_kv", GgufValue::Uint32(10)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(phi4_tokens)),
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(4)),
            (
                "tokenizer.chat_template",
                GgufValue::String("<|im_start|>user<|im_sep|>{{msg}}<|im_end|>"),
            ),
        ];
        let phi4_bytes = create_synthetic_gguf_v3_with_kvs("phi4", 32, &phi4_kvs, &[]);
        let phi4_gguf = GgufFile::parse(&phi4_bytes).expect("parse phi4");
        let phi4_recipe = Phi3Recipe::from_gguf(&phi4_gguf).expect("hydrate phi4");
        let phi4_stop = phi4_recipe
            .extract_stop_metadata(&phi4_gguf)
            .expect("stop metadata phi4");
        assert_eq!(phi4_stop.vocab_size, PHI4_14B_VOCAB_SIZE);
        assert_eq!(phi4_stop.dialect, PhiChatDialect::Phi4ImSep);
        assert_eq!(phi4_stop.im_sep_token_id, Some(3));
        assert_eq!(phi4_stop.im_end_token_id, Some(4));
        let crit4 = phi4_stop.stop_criteria(16, 128);
        assert_eq!(crit4.evaluate(3, 1, 2), Some(StopReason::EosToken(3)));
        assert_eq!(crit4.evaluate(4, 1, 2), Some(StopReason::EosToken(4)));

        let phi4_mini_u32 = u32::try_from(PHI4_MINI_VOCAB_SIZE).expect("fits u32");
        let phi4_mini_kvs = [("phi3.vocab_size", GgufValue::Uint32(phi4_mini_u32))];
        let phi4_mini_bytes = create_synthetic_gguf_v3_with_kvs("phi3", 32, &phi4_mini_kvs, &[]);
        let phi4_mini_gguf = GgufFile::parse(&phi4_mini_bytes).expect("parse phi4 mini");
        let phi4_mini_recipe = Phi3Recipe::from_gguf(&phi4_mini_gguf).expect("phi4 mini");
        assert_eq!(phi4_mini_recipe.vocab_size, PHI4_MINI_VOCAB_SIZE);
    }

    #[test]
    fn test_vocab_bounds_and_logits_mismatch_fail_closed() {
        let bad_vocab_kvs = [
            ("phi3.embedding_length", GgufValue::Uint32(4)),
            ("phi3.vocab_size", GgufValue::Uint32(12)),
        ];
        let bad_vocab_bytes = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &bad_vocab_kvs,
            &[("token_embd.weight", QuantType::F32, &[4, 16])],
        );
        let bad_vocab_gguf = GgufFile::parse(&bad_vocab_bytes).expect("parse mismatch gguf");
        assert!(matches!(
            Phi3Recipe::from_gguf(&bad_vocab_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let untied_diverge_bytes = create_synthetic_gguf_v3_with_kvs(
            "phi3",
            32,
            &[("phi3.embedding_length", GgufValue::Uint32(4))],
            &[
                ("token_embd.weight", QuantType::F32, &[4, 16]),
                ("output.weight", QuantType::F32, &[4, 12]),
            ],
        );
        let untied_gguf = GgufFile::parse(&untied_diverge_bytes).expect("parse untied mismatch");
        assert!(matches!(
            Phi3Recipe::from_gguf(&untied_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let oob_eos_kvs = [
            ("phi3.vocab_size", GgufValue::Uint32(16)),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(16)),
        ];
        let oob_eos_bytes = create_synthetic_gguf_v3_with_kvs("phi3", 32, &oob_eos_kvs, &[]);
        let oob_eos_gguf = GgufFile::parse(&oob_eos_bytes).expect("parse oob eos");
        assert!(matches!(
            Phi3Recipe::from_gguf(&oob_eos_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let mut syn_recipe = Phi3Recipe::new();
        syn_recipe.num_layers = 1;
        syn_recipe.hidden_dim = 4;
        syn_recipe.ffn_dim = 8;
        syn_recipe.n_heads_q = 2;
        syn_recipe.n_heads_kv = 2;
        syn_recipe.head_dim = 2;
        syn_recipe.rotary_dim = Some(2);
        syn_recipe.vocab_size = 8;
        let syn_weights = TestWeights { tensors: vec![] };

        let total_kv = syn_recipe.n_heads_kv * syn_recipe.head_dim;
        let mut kv_small = vec![0.0_f32; 8 * total_kv * 2];
        let mut short_logits = [0.0_f32; 6];
        let mut ctx_short =
            ExecutionContext::new(&[1], 8, &mut short_logits, &mut kv_small).expect("short ctx");
        assert!(matches!(
            syn_recipe.prefill_step(&mut ctx_short, &syn_weights),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let mut kv_large = vec![0.0_f32; 8 * total_kv * 2];
        let mut long_logits = [0.0_f32; 10];
        let mut ctx_long =
            ExecutionContext::new(&[1], 8, &mut long_logits, &mut kv_large).expect("long ctx");
        assert!(matches!(
            syn_recipe.prefill_step(&mut ctx_long, &syn_weights),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_wave3_batched_prefill_chunked_start_pos_kv_and_logits_parity_miri() {
        let mut recipe = build_1layer_recipe(4, 8, 2, 1, 6);
        recipe.original_max_position_embeddings = 2;
        recipe = recipe
            .with_rotary_dim(Some(2))
            .with_rope_scaling(RopeScalingConfig::LongRope {
                short_factor: vec![1.1],
                long_factor: vec![2.3],
                original_max_position_embeddings: 2,
            })
            .with_attn_factor(1.15)
            .with_sliding_window(Some(2));

        let specs = build_full_weight_buffers(4, 8, 4, 2, 6, QuantType::F32, QuantType::F32, true);
        let weights = weights_from_buffers(&specs);
        let prompt = [1_u32, 3, 5, 2];

        let mut logits_batch = [0.0_f32; 6];
        let mut kv_batch = vec![0.0_f32; 6 * 2 * 2];
        let mut ctx_b =
            ExecutionContext::new(&prompt, 6, &mut logits_batch, &mut kv_batch).expect("batch ctx");
        recipe
            .prefill_step(&mut ctx_b, &weights)
            .expect("batched prefill");

        // Chunked prefill across LongRoPE boundary (`orig_max=2`) and sliding window (`2`).
        let mut logits_chunk = [0.0_f32; 6];
        let mut kv_chunk = vec![0.0_f32; 6 * 2 * 2];
        let mut empty_logits: [f32; 0] = [];
        let mut ctx_c0 = ExecutionContext::new_prefill_chunk(
            &prompt[..2],
            6,
            0,
            false,
            &mut empty_logits,
            &mut kv_chunk,
        )
        .expect("chunk 0");
        recipe.prefill_step(&mut ctx_c0, &weights).expect("run c0");
        assert_eq!(ctx_c0.position(), 2);

        let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
            &prompt[2..],
            6,
            2,
            true,
            &mut logits_chunk,
            &mut kv_chunk,
        )
        .expect("chunk 1");
        recipe.prefill_step(&mut ctx_c1, &weights).expect("run c1");
        assert_eq!(ctx_c1.position(), 4);

        // Sequential decode equivalence across all 4 positions.
        let mut logits_seq = [0.0_f32; 6];
        let mut kv_seq = vec![0.0_f32; 6 * 2 * 2];
        let mut ctx_s =
            ExecutionContext::new(&prompt[..1], 6, &mut logits_seq, &mut kv_seq).expect("seq ctx");
        recipe.prefill_step(&mut ctx_s, &weights).expect("p0");
        for &tok in &prompt[1..] {
            ctx_s.prepare_decode(tok).expect("prep");
            recipe.decode_step(&mut ctx_s, &weights).expect("dec");
        }

        for (i, (&b, (&c, &s))) in logits_batch
            .iter()
            .zip(logits_chunk.iter().zip(&logits_seq))
            .enumerate()
        {
            assert!((b - c).abs() < 1e-4, "logit[{i}] batch {b} != chunk {c}");
            assert!((b - s).abs() < 1e-4, "logit[{i}] batch {b} != seq {s}");
        }
        for (i, (&b, (&c, &s))) in kv_batch
            .iter()
            .zip(kv_chunk.iter().zip(&kv_seq))
            .enumerate()
        {
            assert!((b - c).abs() < 1e-4, "kv[{i}] batch {b} != chunk {c}");
            assert!((b - s).abs() < 1e-4, "kv[{i}] batch {b} != seq {s}");
        }
    }

    #[test]
    fn test_phi3_split_matrix_gemm_shared_act_and_rows_miri() {
        const COLS: usize = 8;
        const ROWS: usize = 10; // Crosses REPACK_TILE_ROWS = 8
        const M: usize = 2;

        let w_f32: Vec<f32> = (0..ROWS * COLS)
            .map(|i| pattern_f32(i, 3, 1, 13, 6.0, 0.08))
            .collect();
        let w_bytes: Vec<u8> = w_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tv = TensorView::from_bytes("w", QuantType::F32, &[COLS, ROWS], 0, &w_bytes)
            .expect("tensor view");
        let mat = Phi3SplitMatrix::from_tensor(&tv, ROWS, COLS).expect("split matrix");

        let act: Vec<f32> = (0..M * COLS)
            .map(|i| pattern_f32(i, 5, 2, 11, 5.0, 0.1))
            .collect();
        let mut out_full = vec![0.0_f32; M * ROWS];
        mat.gemm(&mut out_full, &act, M).expect("full gemm");

        #[cfg(feature = "with-kernels")]
        {
            let mut q_act = jero_kernels_rvv::QuantizedActQ8::new();
            q_act.quantize_into(&act, M, COLS).expect("q_act");
            let mut out_q = vec![0.0_f32; M * ROWS];
            mat.gemm_quantized(&mut out_q, &q_act).expect("quant gemm");

            let split = REPACK_TILE_ROWS;
            let mut top = vec![0.0_f32; M * split];
            let mut bot = vec![0.0_f32; M * (ROWS - split)];
            mat.gemm_quantized_rows(&mut top, &q_act, 0..split)
                .expect("top rows");
            mat.gemm_quantized_rows(&mut bot, &q_act, split..ROWS)
                .expect("bot rows");

            for m_idx in 0..M {
                for r in 0..ROWS {
                    let full_v = out_full[m_idx * ROWS + r];
                    let q_v = out_q[m_idx * ROWS + r];
                    let row_v = if r < split {
                        top[m_idx * split + r]
                    } else {
                        bot[m_idx * (ROWS - split) + (r - split)]
                    };
                    assert!((full_v - q_v).abs() < 1e-5);
                    assert!((q_v - row_v).abs() < 1e-5);
                }
            }
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            for m_idx in 0..M {
                let mut single = vec![0.0_f32; ROWS];
                mat.gemv(&mut single, &act[m_idx * COLS..(m_idx + 1) * COLS])
                    .expect("gemv");
                for r in 0..ROWS {
                    assert!((out_full[m_idx * ROWS + r] - single[r]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_wave4_scratch_workspace_decode_and_descriptors_miri() {
        let mut recipe = Phi3Recipe::new();
        recipe.num_layers = 2;
        recipe.hidden_dim = 16;
        recipe.ffn_dim = 20;
        recipe.n_heads_q = 4;
        recipe.n_heads_kv = 2;
        recipe.head_dim = 4;
        recipe.vocab_size = 12;

        let desc0 = Phi3Recipe::layer_descriptor(0);
        let desc31 = Phi3Recipe::layer_descriptor(31);
        assert_eq!(desc0.attn_norm(), "blk.0.attn_norm.weight");
        assert_eq!(desc31.attn_output(), "blk.31.attn_output.weight");
        assert_eq!(
            desc31.post_attention_norm(),
            "blk.31.post_attention_norm.weight"
        );
        assert_eq!(desc31.ffn_down(), "blk.31.ffn_down.weight");
        assert_eq!(recipe.layer_descriptors().len(), 2);

        let needed = recipe.required_decode_scratch_bytes(4);
        let mut scratch = jero_core::ScratchWorkspace::new(needed).expect("scratch workspace");
        let kv_geom = jero_core::KvGeometry::new(
            recipe.num_layers,
            recipe.n_heads_kv,
            recipe.head_dim,
            4,
            None,
        )
        .expect("kv geometry");
        let kv_len = kv_geom.required_f32_len().expect("kv f32 len");
        let empty = TestWeights {
            tensors: Vec::new(),
        };

        let mut logits_with_scratch = vec![0.0_f32; recipe.vocab_size];
        let mut kv_with_scratch = vec![0.0_f32; kv_len];
        let mut logits_fallback = vec![0.0_f32; recipe.vocab_size];
        let mut kv_fallback = vec![0.0_f32; kv_len];

        let mut ctx_scratch =
            ExecutionContext::new(&[], 4, &mut logits_with_scratch, &mut kv_with_scratch)
                .expect("scratch ctx")
                .with_kv_geometry(kv_geom)
                .expect("kv geom")
                .with_scratch(&mut scratch);
        ctx_scratch.prepare_decode(1).expect("prep pos 0");
        recipe
            .decode_step(&mut ctx_scratch, &empty)
            .expect("decode pos 0");
        assert_eq!(
            ctx_scratch.scratch().expect("scratch").used_bytes(),
            0,
            "ScratchScope must rewind cursor on exit"
        );
        let peak0 = ctx_scratch.scratch().expect("scratch").peak_used_bytes();
        assert!(peak0 > 0 && peak0 <= needed);

        ctx_scratch.prepare_decode(3).expect("prep pos 1");
        recipe
            .decode_step(&mut ctx_scratch, &empty)
            .expect("decode pos 1");
        let scratch_ref = ctx_scratch.scratch().expect("scratch");
        assert_eq!(scratch_ref.used_bytes(), 0);
        assert!(scratch_ref.peak_used_bytes() >= peak0);

        let mut ctx_fallback =
            ExecutionContext::new(&[], 4, &mut logits_fallback, &mut kv_fallback)
                .expect("fallback ctx")
                .with_kv_geometry(kv_geom)
                .expect("fallback geom");
        ctx_fallback.prepare_decode(1).expect("fallback prep pos 0");
        recipe
            .decode_step(&mut ctx_fallback, &empty)
            .expect("fallback decode pos 0");
        ctx_fallback.prepare_decode(3).expect("fallback prep pos 1");
        recipe
            .decode_step(&mut ctx_fallback, &empty)
            .expect("fallback decode pos 1");

        for (i, (&a, &b)) in logits_with_scratch.iter().zip(&logits_fallback).enumerate() {
            #[cfg(not(miri))]
            assert_eq!(a.to_bits(), b.to_bits(), "logit[{i}] bits mismatch");
            #[cfg(miri)]
            {
                // Miri intentionally jitters low bits of `libm` transcendentals (`sin`, `cos`, `exp`)
                // across evaluations, so verify tight numeric parity instead of exact bit identity.
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5,
                    "logit[{i}] miri mismatch: {a} vs {b}"
                );
            }
        }

        let mut tiny_scratch = jero_core::ScratchWorkspace::new(128).expect("tiny scratch");
        let mut bad_logits = vec![0.0_f32; recipe.vocab_size];
        let mut bad_kv = vec![0.0_f32; kv_len];
        let mut bad_ctx = ExecutionContext::new(&[], 4, &mut bad_logits, &mut bad_kv)
            .expect("bad ctx")
            .with_kv_geometry(kv_geom)
            .expect("bad geom")
            .with_scratch(&mut tiny_scratch);
        bad_ctx.prepare_decode(0).expect("bad prep");
        let err = recipe
            .decode_step(&mut bad_ctx, &empty)
            .expect_err("undersized ScratchWorkspace must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_wave4_split_matrix_m1_gemv_parallel_miri() {
        const GEMV_ROWS: usize = 12;
        const GEMV_COLS: usize = 8;
        let w_f32: Vec<f32> = (0..GEMV_ROWS * GEMV_COLS)
            .map(|i| pattern_f32(i, 2, 1, 7, 5.0, 0.09))
            .collect();
        let w_bytes: Vec<u8> = w_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tv =
            TensorView::from_bytes("w_m1", QuantType::F32, &[GEMV_COLS, GEMV_ROWS], 0, &w_bytes)
                .expect("tv");
        let split = Phi3SplitMatrix::from_tensor(&tv, GEMV_ROWS, GEMV_COLS).expect("split");
        let x: Vec<f32> = (0..GEMV_COLS)
            .map(|i| pattern_f32(i, 1, 0, 5, 4.0, 0.1))
            .collect();
        let mut out_seq = [0.0_f32; GEMV_ROWS];
        let mut out_par = [0.0_f32; GEMV_ROWS];
        split.gemv(&mut out_seq, &x).expect("seq gemv");
        split.gemv_parallel(&mut out_par, &x).expect("par gemv");
        for i in 0..GEMV_ROWS {
            assert_eq!(out_seq[i].to_bits(), out_par[i].to_bits());
        }
    }
}
