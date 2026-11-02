//! `k3-arch-phi3`: Microsoft Phi architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`ArchRecipe`] for Microsoft Phi-3 and Phi-4 models (`phi3`, `phi4`),
//! supporting sliced fused QKV/MLP tensors and dynamic `LongRoPE` context switching.

pub mod recipe;

pub use recipe::{Phi3Recipe, RopeScalingConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

    struct TestWeights<'a> {
        tensors: Vec<(&'static str, TensorView<'a>)>,
    }

    impl ModelWeights for TestWeights<'_> {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            self.tensors
                .iter()
                .find(|(n, _)| *n == name)
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

    #[test]
    fn test_phi3_recipe_basic() {
        let recipe = Phi3Recipe::new();
        assert_eq!(recipe.architecture_id(), "phi3");
        assert_eq!(recipe.recipe_version(), "1.0.0");
        assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
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

        // Position <= 4096 is short context
        assert!(!recipe.is_long_context(4096));
        let short_factors = recipe.active_rope_factors(4096).expect("factors available");
        assert!((short_factors[0] - 1.0).abs() < 1e-6);

        // Position > 4096 switches to long factor table
        assert!(recipe.is_long_context(4097));
        let long_factors = recipe.active_rope_factors(4097).expect("factors available");
        assert!((long_factors[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row index in test fits in u8"
    )]
    fn test_phi3_prefill_and_decode_step_with_fused_slicing() {
        // Prepare mock fused weights:
        // blk.0.attn_qkv.weight: 32 columns x 6 rows (Q=2 rows, K=2 rows, V=2 rows)
        // Q8_0: 32 cols -> 1 block per row -> 34 bytes per row * 6 = 204 bytes
        let shape_qkv = [32, 6];
        let mut qkv_raw = vec![0x3C_u8; 34 * 6];
        for r in 0..6 {
            let start = r * 34;
            qkv_raw[start..start + 2].copy_from_slice(&[0x00, 0x3C]); // scale 1.0
            qkv_raw[start + 2..start + 34].fill((r as u8) + 1);
        }
        let qkv_view = TensorView::from_bytes(
            "blk.0.attn_qkv.weight",
            QuantType::Q8_0,
            &shape_qkv,
            0,
            &qkv_raw,
        )
        .expect("valid qkv view");

        // blk.0.ffn_up.weight: 32 columns x 4 rows (Gate=2 rows, Up=2 rows)
        let shape_ffn = [32, 4];
        let mut ffn_raw = vec![0x3C_u8; 34 * 4];
        for r in 0..4 {
            let start = r * 34;
            ffn_raw[start..start + 2].copy_from_slice(&[0x00, 0x3C]);
            ffn_raw[start + 2..start + 34].fill((r as u8) + 10);
        }
        let ffn_view = TensorView::from_bytes(
            "blk.0.ffn_up.weight",
            QuantType::Q8_0,
            &shape_ffn,
            0,
            &ffn_raw,
        )
        .expect("valid ffn view");

        let weights = TestWeights {
            tensors: vec![
                ("blk.0.attn_qkv.weight", qkv_view),
                ("blk.0.ffn_up.weight", ffn_view),
            ],
        };

        let recipe = Phi3Recipe::new();
        let prompt = [10_u32, 20, 30];
        let mut logits = [0.0_f32; 16];
        let mut kv_cache = [0.0_f32; 64];

        let mut ctx =
            ExecutionContext::new(&prompt, 16, &mut logits, &mut kv_cache).expect("create context");

        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds with fused slicing");
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits()[9] >= 1.0);

        ctx.prepare_decode(5).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds with fused slicing");
        assert_eq!(ctx.position(), 4);
        assert!(ctx.logits()[3] >= 2.0);
    }
}
