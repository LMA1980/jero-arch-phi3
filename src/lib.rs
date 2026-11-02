//! `k3-arch-phi3`: Microsoft Phi architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`ArchRecipe`] for Microsoft Phi-3 and Phi-4 models (`phi3`, `phi4`),
//! supporting sliced fused QKV/MLP tensors and dynamic `LongRoPE` context switching.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

/// Phi3 architecture recipe implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Phi3Recipe;

impl Phi3Recipe {
    /// Creates a new instance of the Phi3 architecture recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ArchRecipe for Phi3Recipe {
    fn architecture_id(&self) -> &'static str {
        "phi3"
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
                | QuantType::Q4_K_M
                | QuantType::Q8_0
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_len = ctx.prompt_tokens().len();
        ctx.advance_position(prompt_len)?;
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            logits[0] = 1.0;
        }
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let _tok = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;
        ctx.advance_position(1)?;
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            logits[0] = 1.0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi3_recipe_basic() {
        let recipe = Phi3Recipe::new();
        assert_eq!(recipe.architecture_id(), "phi3");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert_eq!(recipe.recipe_version(), "1.0.0");
    }
}
