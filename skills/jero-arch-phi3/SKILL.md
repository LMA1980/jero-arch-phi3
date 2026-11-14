---
name: jero-arch-phi3
description: "Implement and validate Microsoft Phi-3 / Phi-4 ArchRecipe execution, fused QKV and gate-up slicing, and LongRoPE scaling factors in jero-arch-phi3."
---

# `jero-arch-phi3` Fused QKV/Gate-Up Slicing, Batched Prefill GEMM & Metadata Contracts

- Implement `jero_core::ArchRecipe` handling fused `attn_qkv` (`Q || K || V`) and fused `ffn_up` (`gate || up`) tensor slicing without intermediate heap copies where possible, plus `LongRoPE` short/long per-dimension frequency scaling factors.
- Execute `Phi3Recipe::prefill_step` (`prefill_step_with_repack`) across `[M, hidden_dim]` prompt chunks per layer via `QuantizedActQ8::quantize_into` shared once across `(Wq, Wk, Wv)` and once across `(Wgate, Wup)`, dispatching tile-row-sliced `gemm_repacked_rows` on `A100` `XSMTVDotII` (`smt.vmadot`) / `RVV`, batched `LongRoPE` (`apply_rope_with_cos_sin`), and KV-head-partitioned `gqa_causal_attention_prefill_f32` with `ExecutionContext` `start_position` chunking, while projecting only final token `M - 1` through the LM head when `ctx.compute_logits()` is enabled.
- Use `Phi3ModelRepack` (`Phi3RepackCache`) / `Phi3LayerRepack` to pre-split fused `attn_qkv.weight` into independent `(Wq, Wk, Wv)` and `ffn_up.weight` into `(Wgate, Wup)` `128-byte`-aligned `jero_kernels_rvv::RepackedWeight` matrices and cache decoded `f32` `RMSNorm` vectors once at model load (with ABA-safe bounded layer caching in `prefill_step` / `decode_step` under `with-kernels`), preventing `8x16` tile straddling across GQA and non-multiple-of-8 slice boundaries.
- Enforce fail-closed `K3Error::CapabilityMismatch` vocabulary bounds across GGUF metadata, `token_embd.weight` / `output.weight` shapes, special token IDs, and logit buffers, and extract `PhiStopMetadata` (`Phi3End` `<|end|>` vs `Phi4ImSep` `<|im_sep|>` / `<|im_end|>`) for `jero_core::StopCriteria`.
- Preserve portable scalar parity under `--no-default-features` and delegate to `jero-kernels-rvv` when `with-kernels` is enabled.
