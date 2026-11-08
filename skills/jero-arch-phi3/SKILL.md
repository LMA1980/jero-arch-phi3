---
name: jero-arch-phi3
description: "Implement and validate Microsoft Phi-3 / Phi-4 ArchRecipe execution, fused QKV and gate-up slicing, and LongRoPE scaling factors in jero-arch-phi3."
---

# `jero-arch-phi3` Fused QKV/Gate-Up Slicing & LongRoPE Invariants

- Implement `jero_core::ArchRecipe` handling fused `attn_qkv` (`Q || K || V`) and fused `ffn_up` (`gate || up`) tensor slicing without intermediate heap copies where possible, plus `LongRoPE` short/long per-dimension frequency scaling factors.
- Preserve tensor slicing and delegates to `jero-kernels-rvv` when the `with-kernels` feature is enabled.
