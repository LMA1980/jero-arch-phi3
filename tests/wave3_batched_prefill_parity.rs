//! Integration parity tests for Wave 3 batched `M > 1` `Phi3Recipe::prefill_step` (`gemm_repacked_rows`,
//! shared `QuantizedActQ8`, batched `LongRoPE` + causal GQA, `start_position` chunking, and
//! `compute_logits()` skipping).

use jero_arch_phi3::{Phi3Recipe, REPACK_TILE_ROWS, RopeScalingConfig};
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

fn pattern_f32(r: usize, c: usize, seed: usize) -> f32 {
    let bucket = u8::try_from((r * 13 + c * 7 + seed) % 31).expect("modulus fits u8");
    (f32::from(bucket) - 15.0) * 0.025
}

fn encode_matrix(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::new();
    match quant {
        QuantType::F32 => {
            for r in 0..rows {
                for c in 0..cols {
                    out.extend_from_slice(&pattern_f32(r, c, seed).to_le_bytes());
                }
            }
        }
        QuantType::F16 => {
            for r in 0..rows {
                for c in 0..cols {
                    let v = pattern_f32(r, c, seed);
                    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
            }
        }
        QuantType::BF16 => {
            for r in 0..rows {
                for c in 0..cols {
                    let v = pattern_f32(r, c, seed);
                    let bits = u16::try_from(v.to_bits() >> 16).unwrap_or(0);
                    out.extend_from_slice(&bits.to_le_bytes());
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
        other => panic!("unsupported test quant {other:?}"),
    }
    out
}

fn encode_q4_0_q5_0(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::new();
    match quant {
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
        other => panic!("unexpected quant {other:?}"),
    }
    out
}

fn encode_any(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    if matches!(quant, QuantType::Q4_0 | QuantType::Q5_0) {
        encode_q4_0_q5_0(quant, cols, rows, seed)
    } else {
        encode_matrix(quant, cols, rows, seed)
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
            encode_any(QuantType::Q4_0, hidden, vocab, 3),
        ),
        (
            "output_norm.weight".to_owned(),
            QuantType::F16,
            vec![hidden],
            encode_any(QuantType::F16, hidden, 1, 9),
        ),
        (
            "output.weight".to_owned(),
            QuantType::Q8_0,
            vec![hidden, vocab],
            encode_any(QuantType::Q8_0, hidden, vocab, 17),
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
            encode_any(norm_q, hidden, 1, 21 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.attn_qkv.weight"),
            qkv_q,
            vec![hidden, q_dim + 2 * kv_dim],
            encode_any(qkv_q, hidden, q_dim + 2 * kv_dim, 31 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.attn_output.weight"),
            QuantType::Q8_0,
            vec![q_dim, hidden],
            encode_any(QuantType::Q8_0, q_dim, hidden, 41 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_norm.weight"),
            QuantType::BF16,
            vec![hidden],
            encode_any(QuantType::BF16, hidden, 1, 51 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_up.weight"),
            up_q,
            vec![hidden, 2 * ffn_dim],
            encode_any(up_q, hidden, 2 * ffn_dim, 61 + layer),
        ));
        bufs.push((
            format!("blk.{layer}.ffn_down.weight"),
            down_q,
            vec![ffn_dim, hidden],
            encode_any(down_q, ffn_dim, hidden, 71 + layer),
        ));
    }
    bufs
}

fn assert_slice_close(lhs: &[f32], rhs: &[f32], tol: f32, label: &str) {
    let diff = lhs
        .iter()
        .zip(rhs)
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(diff < tol, "{label} diff={diff} >= {tol}");
}

fn run_sequential_repack_prompt(
    recipe: &Phi3Recipe,
    weights: &dyn ModelWeights,
    cache: &jero_arch_phi3::Phi3ModelRepack,
    prompt: &[u32],
    kv_buf: &mut [f32],
    logits_out: &mut [f32],
) {
    let mut ctx = ExecutionContext::new(&prompt[..1], 16, logits_out, kv_buf).expect("s0 ctx");
    recipe
        .prefill_step_with_repack(&mut ctx, weights, cache)
        .expect("s0 prefill");
    for &tok in &prompt[1..] {
        ctx.prepare_decode(tok).expect("prep decode");
        recipe
            .decode_step_with_repack(&mut ctx, weights, cache)
            .expect("decode");
    }
}

fn views_from_specs(specs: &[(String, QuantType, Vec<usize>, Vec<u8>)]) -> TestWeights<'_> {
    let tensors = specs
        .iter()
        .map(|(n, q, s, r)| {
            (
                n.clone(),
                TensorView::from_bytes(n, *q, s, 0, r).expect("view"),
            )
        })
        .collect();
    TestWeights { tensors }
}

fn build_gqa40_2layer_recipe() -> Phi3Recipe {
    let mut recipe = Phi3Recipe::new_phi4();
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 40;
    recipe.n_heads_kv = 10;
    recipe.head_dim = 8;
    recipe.rotary_dim = Some(8);
    recipe.vocab_size = 16;
    recipe.rope_scaling = RopeScalingConfig::Linear { factor: 1.5 };
    recipe
}

#[test]
fn test_wave3_batched_vs_chunked_vs_sequential_repack_equivalence() {
    const HIDDEN: usize = 256;
    const Q_DIM: usize = 40 * 8;
    const KV_DIM: usize = 10 * 8;
    const FFN_DIM: usize = 256;
    const VOCAB: usize = 16;

    let recipe = build_gqa40_2layer_recipe();
    let bufs = build_mixed_quant_2layer_specs(HIDDEN, Q_DIM, KV_DIM, FFN_DIM, VOCAB);
    let weights = views_from_specs(&bufs);
    let cache = recipe.build_repack_cache(&weights).expect("cache");

    let prompt = [1_u32, 4, 7, 2, 11];
    let kv_len = 2 * 16 * KV_DIM * 2;
    let mut logits_batch = [0.0_f32; VOCAB];
    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut ctx_batch =
        ExecutionContext::new(&prompt, 16, &mut logits_batch, &mut kv_batch).expect("b");
    recipe
        .prefill_step_with_repack(&mut ctx_batch, &weights, &cache)
        .expect("batch prefill");

    let mut logits_uncached = [0.0_f32; VOCAB];
    let mut kv_uncached = vec![0.0_f32; kv_len];
    let mut ctx_uncached =
        ExecutionContext::new(&prompt, 16, &mut logits_uncached, &mut kv_uncached).expect("u");
    recipe
        .prefill_step(&mut ctx_uncached, &weights)
        .expect("uncached prefill");

    let mut logits_seq = [0.0_f32; VOCAB];
    let mut kv_seq = vec![0.0_f32; kv_len];
    run_sequential_repack_prompt(
        &recipe,
        &weights,
        &cache,
        &prompt,
        &mut kv_seq,
        &mut logits_seq,
    );
    assert_slice_close(&logits_batch, &logits_seq, 1e-4, "batch vs seq logits");
    assert_slice_close(&kv_batch, &kv_seq, 1e-4, "batch vs seq kv");

    let sentinel = -777.0_f32;
    let mut logits_chunk = [sentinel; VOCAB];
    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut ctx_c0 = ExecutionContext::new_prefill_chunk(
        &prompt[..2],
        16,
        0,
        false,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .expect("c0");
    recipe
        .prefill_step_with_repack(&mut ctx_c0, &weights, &cache)
        .expect("run c0");
    assert!(
        logits_chunk
            .iter()
            .all(|&v| (v - sentinel).abs() < f32::EPSILON)
    );

    let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
        &prompt[2..],
        16,
        2,
        true,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .expect("c1");
    recipe
        .prefill_step_with_repack(&mut ctx_c1, &weights, &cache)
        .expect("run c1");
    assert_slice_close(&logits_batch, &logits_chunk, 1e-4, "batch vs chunk logits");
    assert_slice_close(&kv_batch, &kv_chunk, 1e-4, "batch vs chunk kv");

    let _ = REPACK_TILE_ROWS;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let mut logits_par = [0.0_f32; VOCAB];
    let mut kv_par = vec![0.0_f32; kv_len];
    pool.install(|| {
        let mut ctx_par =
            ExecutionContext::new(&prompt, 16, &mut logits_par, &mut kv_par).expect("p");
        recipe
            .prefill_step_with_repack(&mut ctx_par, &weights, &cache)
            .expect("par prefill");
    });
    assert_slice_close(
        &logits_batch,
        &logits_uncached,
        1e-4,
        "batch vs uncached logits",
    );
    assert_slice_close(&kv_batch, &kv_uncached, 1e-4, "batch vs uncached kv");
    assert_slice_close(&logits_batch, &logits_par, 1e-5, "par vs batch logits");
    assert_slice_close(&kv_batch, &kv_par, 1e-5, "par vs batch kv");
}

#[test]
fn test_wave3_longrope_switching_and_sliding_window_chunked_parity() {
    const HIDDEN: usize = 256;
    const Q_DIM: usize = 24 * 8;
    const KV_DIM: usize = 8 * 8;
    const FFN_DIM: usize = 256;
    const VOCAB: usize = 16;

    let mut recipe = Phi3Recipe::new_phi4_mini();
    recipe.num_layers = 2;
    recipe.hidden_dim = HIDDEN;
    recipe.ffn_dim = FFN_DIM;
    recipe.n_heads_q = 24;
    recipe.n_heads_kv = 8;
    recipe.head_dim = 8;
    recipe.rotary_dim = Some(8);
    recipe.vocab_size = VOCAB;
    recipe.original_max_position_embeddings = 3;
    recipe = recipe
        .with_rope_scaling(RopeScalingConfig::LongRope {
            short_factor: vec![1.0, 1.1, 1.2, 1.3],
            long_factor: vec![2.0, 2.25, 2.5, 2.75],
            original_max_position_embeddings: 3,
        })
        .with_attn_factor(1.125)
        .with_sliding_window(Some(3));

    let bufs = build_mixed_quant_2layer_specs(HIDDEN, Q_DIM, KV_DIM, FFN_DIM, VOCAB);
    let weights = views_from_specs(&bufs);
    let cache = recipe.build_repack_cache(&weights).expect("cache");
    let prompt = [2_u32, 5, 8, 1, 9, 4];
    let kv_len = 2 * 16 * KV_DIM * 2;

    let mut logits_batch = [0.0_f32; VOCAB];
    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut ctx_batch =
        ExecutionContext::new(&prompt, 16, &mut logits_batch, &mut kv_batch).expect("b");
    recipe
        .prefill_step(&mut ctx_batch, &weights)
        .expect("batch prefill");

    let mut logits_chunk = [0.0_f32; VOCAB];
    let mut kv_chunk = vec![0.0_f32; kv_len];
    for (start, compute_logits) in [(0_usize, false), (2, false), (4, true)] {
        let mut empty_logits: [f32; 0] = [];
        let slot: &mut [f32] = if compute_logits {
            &mut logits_chunk
        } else {
            &mut empty_logits
        };
        let mut ctx_c = ExecutionContext::new_prefill_chunk(
            &prompt[start..start + 2],
            16,
            start,
            compute_logits,
            slot,
            &mut kv_chunk,
        )
        .expect("c");
        recipe.prefill_step(&mut ctx_c, &weights).expect("chunk");
        assert_eq!(ctx_c.position(), start + 2);
    }

    let mut logits_seq = [0.0_f32; VOCAB];
    let mut kv_seq = vec![0.0_f32; kv_len];
    run_sequential_repack_prompt(
        &recipe,
        &weights,
        &cache,
        &prompt,
        &mut kv_seq,
        &mut logits_seq,
    );
    assert_slice_close(
        &logits_batch,
        &logits_chunk,
        1e-4,
        "chunk LongRoPE+SWA logits",
    );
    assert_slice_close(&kv_batch, &kv_chunk, 1e-4, "chunk LongRoPE+SWA kv");
    assert_slice_close(&logits_batch, &logits_seq, 1e-4, "seq LongRoPE+SWA logits");
    assert_slice_close(&kv_batch, &kv_seq, 1e-4, "seq LongRoPE+SWA kv");
}

#[test]
fn test_wave3_prefill_chunk_bounds_and_logits_validation() {
    let mut recipe = Phi3Recipe::new();
    recipe.num_layers = 1;
    recipe.hidden_dim = 8;
    recipe.ffn_dim = 16;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 2;
    recipe.rotary_dim = Some(2);
    recipe.vocab_size = 8;

    let empty = TestWeights { tensors: vec![] };
    let total_kv = recipe.n_heads_kv * recipe.head_dim;

    // Wrong logits length succeeds only when `compute_logits = false`, fails closed when `true`.
    let mut empty_logits: [f32; 0] = [];
    let mut kv = vec![0.0_f32; 8 * total_kv * 2];
    let mut ctx_skip =
        ExecutionContext::new_prefill_chunk(&[1, 2], 8, 0, false, &mut empty_logits, &mut kv)
            .expect("skip ctx");
    recipe
        .prefill_step(&mut ctx_skip, &empty)
        .expect("skip logits passes");

    let mut bad_logits = [0.0_f32; 5];
    let mut ctx_bad_logits =
        ExecutionContext::new_prefill_chunk(&[3], 8, 2, true, &mut bad_logits, &mut kv)
            .expect("bad logits ctx");
    assert!(matches!(
        recipe.prefill_step(&mut ctx_bad_logits, &empty),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    // Context bounds violation on decode chunk overflow (`start_position + M > n_ctx`).
    assert!(matches!(
        ExecutionContext::new_prefill_chunk(&[1, 2, 3], 8, 6, false, &mut empty_logits, &mut kv),
        Err(K3Error::ContextLengthExceeded { .. })
    ));
}
