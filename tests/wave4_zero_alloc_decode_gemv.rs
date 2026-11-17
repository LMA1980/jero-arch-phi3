//! Wave 4 verification tests for zero-allocation steady-state `Phi3Recipe::decode_step`
//! (`ScratchWorkspace` + pre-formatted `Phi3LayerDescriptor`) and deterministic row-parallel
//! `M = 1` GEMV across `1`, `4`, and `8` `WorkerPool` threads including `32k`–`200k` LM heads.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use jero_arch_phi3::{
    PHI4_MINI_VOCAB_SIZE, PHI35_PADDED_VOCAB_SIZE, Phi3LayerDescriptor, Phi3ModelRepack,
    Phi3Recipe, Phi3SplitMatrix, REPACK_TILE_ROWS,
};
use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, KvGeometry, ModelWeights, QuantType, ScratchWorkspace,
    TensorView,
};

struct CountingAllocator;

static MEASURE_LOCK: Mutex<()> = Mutex::new(());
static ACTIVE_TRACKING_GEN: AtomicU64 = AtomicU64::new(0);
static NEXT_TRACKING_GEN: AtomicU64 = AtomicU64::new(1);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACKED_THREAD_STATE: Cell<(u64, Option<ThreadId>)> = const { Cell::new((0, None)) };
}

fn register_current_thread(tracking_gen: u64) -> ThreadId {
    let tid = std::thread::current().id();
    TRACKED_THREAD_STATE.with(|state| state.set((tracking_gen, Some(tid))));
    tid
}

fn clear_current_thread() {
    let _ = TRACKED_THREAD_STATE.try_with(|state| state.set((0, None)));
}

fn should_track_current_thread() -> bool {
    let active_gen = ACTIVE_TRACKING_GEN.load(Ordering::SeqCst);
    if active_gen == 0 {
        return false;
    }
    TRACKED_THREAD_STATE
        .try_with(|state| {
            let (thread_gen, thread_id) = state.get();
            thread_gen == active_gen && thread_id.is_some()
        })
        .unwrap_or(false)
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: Every allocator method delegates strictly to `std::alloc::System` with identical
// `Layout` and pointer provenance invariants, updating atomic counters only when the current
// thread's TLS tracking generation and `ThreadId` match `ACTIVE_TRACKING_GEN`. Because TLS uses
// `const`-initialized `!Drop` storage and is populated outside `GlobalAlloc`, unregistered
// `libtest`, coverage-runtime, and background threads are ignored without re-entering `GlobalAlloc`.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if should_track_current_thread() {
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: `layout` satisfies `GlobalAlloc::alloc` preconditions; delegating to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if should_track_current_thread() {
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: `layout` satisfies `GlobalAlloc::alloc_zeroed` preconditions.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` originated from `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if should_track_current_thread() {
            ALLOC_BYTES.fetch_add(new_size, Ordering::SeqCst);
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: `ptr` and `layout` originated from `System`, and `new_size` is valid.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

struct TrackingGuard;

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        ACTIVE_TRACKING_GEN.store(0, Ordering::SeqCst);
        clear_current_thread();
    }
}

fn measure_allocs<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let _lock = MEASURE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tracking_gen = NEXT_TRACKING_GEN.fetch_add(1, Ordering::SeqCst);
    let _active_tid = register_current_thread(tracking_gen);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    let guard = TrackingGuard;
    ACTIVE_TRACKING_GEN.store(tracking_gen, Ordering::SeqCst);
    let result = f();
    drop(guard);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    (result, bytes, count)
}

fn measure_allocs_in_pool<R: Send>(
    pool: &rayon::ThreadPool,
    f: impl FnOnce() -> R + Send,
) -> (R, usize, usize) {
    let _lock = MEASURE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tracking_gen = NEXT_TRACKING_GEN.fetch_add(1, Ordering::SeqCst);
    let registered = pool.broadcast(|_| register_current_thread(tracking_gen));
    assert_eq!(registered.len(), pool.current_num_threads());
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    let guard = TrackingGuard;
    ACTIVE_TRACKING_GEN.store(tracking_gen, Ordering::SeqCst);
    let result = pool.install(f);
    drop(guard);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    pool.broadcast(|_| clear_current_thread());
    (result, bytes, count)
}

struct OwnedWeights {
    blobs: Vec<(String, QuantType, Vec<usize>, Vec<u8>)>,
}

struct OwnedWeightsView<'a> {
    tensors: Vec<(String, TensorView<'a>)>,
}

impl<'a> OwnedWeightsView<'a> {
    fn from_owned(owned: &'a OwnedWeights) -> Self {
        let tensors = owned
            .blobs
            .iter()
            .map(|(name, quant, shape, raw)| {
                let tv = TensorView::from_bytes(name, *quant, shape, 0, raw)
                    .expect("valid owned tensor");
                (name.clone(), tv)
            })
            .collect();
        Self { tensors }
    }
}

impl ModelWeights for OwnedWeightsView<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, tv)| *tv)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor not found".to_owned(),
            })
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n == name)
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.iter().map(|(_, tv)| tv.byte_len()).sum()
    }
}

struct EmptyWeights;

impl ModelWeights for EmptyWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "empty synthetic weights".to_owned(),
        })
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        false
    }

    fn tensor_count(&self) -> usize {
        0
    }

    fn total_weight_bytes(&self) -> usize {
        0
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
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

fn f32_to_bf16_bits(val: f32) -> u16 {
    u16::try_from(val.to_bits() >> 16).unwrap_or(0)
}

fn pattern_val(r: usize, c: usize, seed: usize) -> f32 {
    let bucket = u8::try_from((r * 17 + c * 11 + seed * 5) % 29).expect("modulus fits u8");
    (f32::from(bucket) - 14.0) * 0.02
}

fn encode_matrix_fast(quant: QuantType, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let pattern_rows = rows.min(32);
    let mut pattern = Vec::new();
    match quant {
        QuantType::F32 => {
            for r in 0..pattern_rows {
                for c in 0..cols {
                    pattern.extend_from_slice(&pattern_val(r, c, seed).to_le_bytes());
                }
            }
        }
        QuantType::F16 => {
            for r in 0..pattern_rows {
                for c in 0..cols {
                    pattern
                        .extend_from_slice(&f32_to_f16_bits(pattern_val(r, c, seed)).to_le_bytes());
                }
            }
        }
        QuantType::BF16 => {
            for r in 0..pattern_rows {
                for c in 0..cols {
                    pattern.extend_from_slice(
                        &f32_to_bf16_bits(pattern_val(r, c, seed)).to_le_bytes(),
                    );
                }
            }
        }
        QuantType::Q8_0 => {
            assert_eq!(cols % 32, 0);
            for r in 0..pattern_rows {
                for blk in 0..(cols / 32) {
                    pattern.extend_from_slice(&f32_to_f16_bits(0.05).to_le_bytes());
                    for k in 0..32 {
                        let q =
                            i8::try_from((r * 5 + blk * 3 + k + seed) % 15).map_or(0, |v| v - 7);
                        pattern.push(q.to_ne_bytes()[0]);
                    }
                }
            }
        }
        QuantType::Q4_0 => {
            assert_eq!(cols % 32, 0);
            for r in 0..pattern_rows {
                for blk in 0..(cols / 32) {
                    pattern.extend_from_slice(&f32_to_f16_bits(0.08).to_le_bytes());
                    for k in 0..16 {
                        let lo = u8::try_from((r + blk + k + seed) % 16).unwrap_or(0);
                        let hi = u8::try_from((r * 3 + blk + k + seed) % 16).unwrap_or(0);
                        pattern.push(lo | (hi << 4));
                    }
                }
            }
        }
        QuantType::Q4_K_M => {
            assert_eq!(cols % 256, 0);
            for r in 0..pattern_rows {
                for sb in 0..(cols / 256) {
                    let mut blk = [0_u8; 144];
                    blk[0..2].copy_from_slice(&f32_to_f16_bits(0.125).to_le_bytes());
                    blk[2..4].copy_from_slice(&f32_to_f16_bits(0.03125).to_le_bytes());
                    for (idx, byte) in blk[4..16].iter_mut().enumerate() {
                        *byte = u8::try_from(((r + sb + idx + seed) % 15) + 1).unwrap_or(1);
                    }
                    for (idx, byte) in blk[16..144].iter_mut().enumerate() {
                        let lo = u8::try_from((r + sb * 5 + idx + seed) % 16).unwrap_or(0);
                        let hi = u8::try_from((r * 2 + sb + idx * 3 + seed) % 16).unwrap_or(0);
                        *byte = lo | (hi << 4);
                    }
                    pattern.extend_from_slice(&blk);
                }
            }
        }
        other => panic!("unsupported test matrix quant {other:?}"),
    }

    if rows <= pattern_rows {
        return pattern;
    }
    let row_bytes = pattern.len() / pattern_rows;
    let mut full = Vec::with_capacity(rows * row_bytes);
    let full_chunks = rows / pattern_rows;
    for _ in 0..full_chunks {
        full.extend_from_slice(&pattern);
    }
    let rem_rows = rows % pattern_rows;
    if rem_rows > 0 {
        full.extend_from_slice(&pattern[..rem_rows * row_bytes]);
    }
    full
}

fn encode_norm_f32(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .flat_map(|i| {
            (0.9 + f32::from(u8::try_from((i + seed) % 7).unwrap_or(0)) * 0.03).to_le_bytes()
        })
        .collect()
}

fn push_layer_mixed_weights(
    blobs: &mut Vec<(String, QuantType, Vec<usize>, Vec<u8>)>,
    recipe: &Phi3Recipe,
    layer: usize,
) {
    let h = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let qkv_rows = total_q + 2 * total_kv;
    let ffn = recipe.ffn_dim;
    let desc = Phi3LayerDescriptor::for_layer(layer);

    blobs.push((
        desc.attn_norm().to_owned(),
        QuantType::F32,
        vec![h],
        encode_norm_f32(h, 10 + layer),
    ));
    if layer % 2 == 0 {
        blobs.push((
            desc.attn_qkv().to_owned(),
            QuantType::Q4_K_M,
            vec![h, qkv_rows],
            encode_matrix_fast(QuantType::Q4_K_M, h, qkv_rows, 20 + layer),
        ));
        blobs.push((
            desc.ffn_norm().to_owned(),
            QuantType::F32,
            vec![h],
            encode_norm_f32(h, 30 + layer),
        ));
        blobs.push((
            desc.ffn_up().to_owned(),
            QuantType::Q8_0,
            vec![h, 2 * ffn],
            encode_matrix_fast(QuantType::Q8_0, h, 2 * ffn, 40 + layer),
        ));
    } else {
        blobs.push((
            desc.attn_q().to_owned(),
            QuantType::Q4_0,
            vec![h, total_q],
            encode_matrix_fast(QuantType::Q4_0, h, total_q, 21 + layer),
        ));
        blobs.push((
            desc.attn_k().to_owned(),
            QuantType::Q8_0,
            vec![h, total_kv],
            encode_matrix_fast(QuantType::Q8_0, h, total_kv, 22 + layer),
        ));
        blobs.push((
            desc.attn_v().to_owned(),
            QuantType::Q4_K_M,
            vec![h, total_kv],
            encode_matrix_fast(QuantType::Q4_K_M, h, total_kv, 23 + layer),
        ));
        blobs.push((
            desc.post_attention_norm().to_owned(),
            QuantType::F32,
            vec![h],
            encode_norm_f32(h, 31 + layer),
        ));
        blobs.push((
            desc.ffn_gate().to_owned(),
            QuantType::Q4_0,
            vec![h, ffn],
            encode_matrix_fast(QuantType::Q4_0, h, ffn, 41 + layer),
        ));
        blobs.push((
            desc.ffn_up().to_owned(),
            QuantType::Q4_K_M,
            vec![h, ffn],
            encode_matrix_fast(QuantType::Q4_K_M, h, ffn, 42 + layer),
        ));
    }
    blobs.push((
        desc.attn_output().to_owned(),
        QuantType::Q4_K_M,
        vec![total_q, h],
        encode_matrix_fast(QuantType::Q4_K_M, total_q, h, 50 + layer),
    ));
    blobs.push((
        desc.ffn_down().to_owned(),
        QuantType::Q4_0,
        vec![ffn, h],
        encode_matrix_fast(QuantType::Q4_0, ffn, h, 60 + layer),
    ));
}

fn build_authentic_mixed_phi_weights(
    recipe: &Phi3Recipe,
    lm_quant: QuantType,
    tied: bool,
) -> OwnedWeights {
    let h = recipe.hidden_dim;
    let mut blobs = vec![
        (
            "token_embd.weight".to_owned(),
            lm_quant,
            vec![h, recipe.vocab_size],
            encode_matrix_fast(lm_quant, h, recipe.vocab_size, 1),
        ),
        (
            "output_norm.weight".to_owned(),
            QuantType::F32,
            vec![h],
            encode_norm_f32(h, 2),
        ),
    ];
    if !tied {
        blobs.push((
            "output.weight".to_owned(),
            lm_quant,
            vec![h, recipe.vocab_size],
            encode_matrix_fast(lm_quant, h, recipe.vocab_size, 99),
        ));
    }
    for layer in 0..recipe.num_layers {
        push_layer_mixed_weights(&mut blobs, recipe, layer);
    }
    OwnedWeights { blobs }
}

fn build_recipe(
    hidden: usize,
    ffn: usize,
    h_q: usize,
    h_kv: usize,
    head_dim: usize,
    vocab: usize,
    layers: usize,
) -> Phi3Recipe {
    let mut r = Phi3Recipe::new();
    r.num_layers = layers;
    r.hidden_dim = hidden;
    r.ffn_dim = ffn;
    r.n_heads_q = h_q;
    r.n_heads_kv = h_kv;
    r.head_dim = head_dim;
    r.vocab_size = vocab;
    r
}

fn check_layer_descriptors_and_synthetic_decode_zero_alloc() {
    let mut total = 0_usize;
    let ((), desc_bytes, desc_count) = measure_allocs(|| {
        for layer in 0..48 {
            let d = Phi3LayerDescriptor::for_layer(layer);
            total += d.attn_norm().len()
                + d.ffn_norm().len()
                + d.post_attention_norm().len()
                + d.attn_qkv().len()
                + d.attn_q().len()
                + d.attn_k().len()
                + d.attn_v().len()
                + d.attn_output().len()
                + d.ffn_up().len()
                + d.ffn_gate().len()
                + d.ffn_down().len();
        }
    });
    assert!(total > 0);
    assert_eq!(
        desc_bytes, 0,
        "Phi3LayerDescriptor must allocate 0 heap bytes"
    );
    assert_eq!(desc_count, 0);

    let recipe = build_recipe(64, 128, 4, 2, 16, 128, 2);
    let max_seq = 8;
    let needed = recipe.required_decode_scratch_bytes(max_seq);
    let mut scratch = ScratchWorkspace::new(needed).expect("scratch workspace");
    let kv_geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        max_seq,
        None,
    )
    .expect("kv geometry");
    let kv_len = kv_geom.required_f32_len().expect("kv len");
    let mut logits = vec![0.0_f32; recipe.vocab_size];
    let mut kv = vec![0.0_f32; kv_len];
    let mut ctx = ExecutionContext::new(&[], max_seq, &mut logits, &mut kv)
        .expect("ctx")
        .with_kv_geometry(kv_geom)
        .expect("geom")
        .with_scratch(&mut scratch);

    ctx.prepare_decode(1).expect("prep warmup");
    recipe.decode_step(&mut ctx, &EmptyWeights).expect("warmup");
    for pos in 1..4_usize {
        ctx.prepare_decode(u32::try_from(pos * 3).unwrap_or(0))
            .expect("prep");
        let (res, bytes, count) = measure_allocs(|| recipe.decode_step(&mut ctx, &EmptyWeights));
        res.expect("steady-state synthetic decode");
        assert_eq!(bytes, 0, "pos {pos} allocated {bytes} heap bytes");
        assert_eq!(count, 0, "pos {pos} made {count} heap allocations");
        assert_eq!(ctx.scratch().expect("scratch").used_bytes(), 0);
    }
}

fn check_authentic_decode_zero_alloc_steady_state() {
    // 256 hidden, ffn=512 (divisible by 32 & 256 so Q4_0/Q8_0/Q4_K_M work cleanly in any role)
    let recipe = build_recipe(256, 512, 8, 2, 32, 256, 2);
    let owned = build_authentic_mixed_phi_weights(&recipe, QuantType::Q4_K_M, false);
    let weights = OwnedWeightsView::from_owned(&owned);
    let repack_cache = Phi3ModelRepack::from_weights(&recipe, &weights).expect("repack cache");

    let max_seq = 8;
    let needed = recipe.required_decode_scratch_bytes(max_seq);
    let kv_geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        max_seq,
        None,
    )
    .expect("kv geometry");
    let kv_len = kv_geom.required_f32_len().expect("kv len");

    for use_repack_cache in [false, true] {
        let mut scratch = ScratchWorkspace::new(needed).expect("scratch");
        let mut logits = vec![0.0_f32; recipe.vocab_size];
        let mut kv = vec![0.0_f32; kv_len];
        let mut ctx = ExecutionContext::new(&[], max_seq, &mut logits, &mut kv)
            .expect("ctx")
            .with_kv_geometry(kv_geom)
            .expect("geom")
            .with_scratch(&mut scratch);

        // Warmup token initializes global/layer repack caches & caches one-time TLS state.
        ctx.prepare_decode(5).expect("prep warmup");
        if use_repack_cache {
            recipe
                .decode_step_with_repack(&mut ctx, &weights, &repack_cache)
                .expect("warmup repack decode");
        } else {
            recipe
                .decode_step(&mut ctx, &weights)
                .expect("warmup decode");
        }

        for step in 1..4_usize {
            ctx.prepare_decode(u32::try_from((step * 19) % recipe.vocab_size).unwrap_or(0))
                .expect("prep step");
            let (res, bytes, count) = measure_allocs(|| {
                if use_repack_cache {
                    recipe.decode_step_with_repack(&mut ctx, &weights, &repack_cache)
                } else {
                    recipe.decode_step(&mut ctx, &weights)
                }
            });
            res.expect("steady-state decode");
            assert_eq!(
                bytes, 0,
                "steady-state decode allocated {bytes} heap bytes (repack_cache={use_repack_cache}, step={step})"
            );
            assert_eq!(count, 0);
            assert_eq!(ctx.scratch().expect("scratch").used_bytes(), 0);
        }
    }

    let pool = build_warmed_pool(4);
    let mut scratch = ScratchWorkspace::new(needed).expect("scratch");
    let mut logits = vec![0.0_f32; recipe.vocab_size];
    let mut kv = vec![0.0_f32; kv_len];
    pool.install(|| {
        let mut ctx = ExecutionContext::new(&[], max_seq, &mut logits, &mut kv)
            .expect("ctx")
            .with_kv_geometry(kv_geom)
            .expect("geom")
            .with_scratch(&mut scratch);
        ctx.prepare_decode(5).expect("prep pool warmup");
        recipe
            .decode_step_with_repack(&mut ctx, &weights, &repack_cache)
            .expect("pool warmup repack decode");
        ctx.prepare_decode(17).expect("prep pool steady-state");
        let (res, bytes, count) = measure_allocs_in_pool(&pool, || {
            recipe.decode_step_with_repack(&mut ctx, &weights, &repack_cache)
        });
        res.expect("pool steady-state repack decode");
        assert_eq!(
            bytes, 0,
            "parallel repack decode allocated {bytes} heap bytes"
        );
        assert_eq!(count, 0);
        assert_eq!(ctx.scratch().expect("scratch").used_bytes(), 0);
    });
}

fn check_deterministic_1_4_8_worker_decode_parity() {
    // Include non-multiple-of-8 slice bounds on Q/K/V and FFN (`256` total_q=256, `head_dim=32`,
    // `ffn=512`; split M=1 matrices cover both aligned and tail row partitions).
    let recipe = build_recipe(256, 512, 8, 2, 32, 128, 2);
    let owned = build_authentic_mixed_phi_weights(&recipe, QuantType::Q4_K_M, false);
    let weights = OwnedWeightsView::from_owned(&owned);
    let repack_cache = Phi3ModelRepack::from_weights(&recipe, &weights).expect("repack cache");

    let run_decode_with_workers = |num_workers: usize, use_repack: bool| -> Vec<u32> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("worker pool");
        let max_seq = 8;
        let needed = recipe.required_decode_scratch_bytes(max_seq);
        let mut scratch = ScratchWorkspace::new(needed).expect("scratch");
        let kv_geom = KvGeometry::new(
            recipe.num_layers,
            recipe.n_heads_kv,
            recipe.head_dim,
            max_seq,
            None,
        )
        .expect("kv geometry");
        let kv_len = kv_geom.required_f32_len().expect("kv len");
        let mut logits = vec![0.0_f32; recipe.vocab_size];
        let mut kv = vec![0.0_f32; kv_len];
        pool.install(|| {
            let mut ctx = ExecutionContext::new(&[], max_seq, &mut logits, &mut kv)
                .expect("ctx")
                .with_kv_geometry(kv_geom)
                .expect("geom")
                .with_scratch(&mut scratch);
            for tok in [7_u32, 23_u32, 61_u32] {
                ctx.prepare_decode(tok).expect("prep tok");
                if use_repack {
                    recipe
                        .decode_step_with_repack(&mut ctx, &weights, &repack_cache)
                        .expect("decode_step_with_repack");
                } else {
                    recipe.decode_step(&mut ctx, &weights).expect("decode_step");
                }
            }
        });
        logits.into_iter().map(f32::to_bits).collect()
    };

    for use_repack in [false, true] {
        let bits_1 = run_decode_with_workers(1, use_repack);
        let bits_4 = run_decode_with_workers(4, use_repack);
        let bits_8 = run_decode_with_workers(8, use_repack);
        assert_eq!(
            bits_1, bits_4,
            "1-worker vs 4-worker decode logits must be bit-identical (repack={use_repack})"
        );
        assert_eq!(
            bits_1, bits_8,
            "1-worker vs 8-worker decode logits must be bit-identical (repack={use_repack})"
        );
        assert!(bits_1.iter().any(|&b| b != 0 && b != 0x8000_0000));
    }
}

fn build_warmed_pool(workers: usize) -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("worker pool");
    pool.broadcast(|_| {
        rayon::join(|| (), || ());
    });
    pool.install(|| {});
    // Allow every worker thread to finish spinning and enter cold `Condvar::wait` once so
    // `std::sync::Condvar` initializes its lazy `OnceBox` outside `measure_allocs`.
    std::thread::sleep(std::time::Duration::from_millis(15));
    pool
}

fn check_200k_lm_head_parallel_decode_parity_and_zero_alloc() {
    for vocab_size in [PHI35_PADDED_VOCAB_SIZE, PHI4_MINI_VOCAB_SIZE] {
        let lm_recipe = build_recipe(256, 256, 8, 2, 32, vocab_size, 1);
        let lm_owned = build_authentic_mixed_phi_weights(&lm_recipe, QuantType::Q4_K_M, true);
        let lm_weights = OwnedWeightsView::from_owned(&lm_owned);
        let lm_geom = KvGeometry::new(1, 2, 32, 4, None).expect("lm kv geometry");
        let lm_kv_len = lm_geom.required_f32_len().expect("lm kv len");
        let lm_needed = lm_recipe.required_decode_scratch_bytes(4);

        let mut runs: Vec<Vec<u32>> = Vec::new();
        for workers in [1_usize, 4_usize, 8_usize] {
            let pool = build_warmed_pool(workers);
            let mut scratch = ScratchWorkspace::new(lm_needed).expect("scratch");
            let mut logits = vec![0.0_f32; vocab_size];
            let mut kv = vec![0.0_f32; lm_kv_len];
            pool.install(|| {
                let mut ctx = ExecutionContext::new(&[], 4, &mut logits, &mut kv)
                    .expect("lm ctx")
                    .with_kv_geometry(lm_geom)
                    .expect("lm geom")
                    .with_scratch(&mut scratch);
                for warmup_tok in [11_u32, 13_u32] {
                    ctx.prepare_decode(warmup_tok).expect("prep warmup");
                    lm_recipe
                        .decode_step(&mut ctx, &lm_weights)
                        .expect("warmup large-vocab decode");
                }

                ctx.prepare_decode(42).expect("prep step");
                let (res, bytes, count) =
                    measure_allocs_in_pool(&pool, || lm_recipe.decode_step(&mut ctx, &lm_weights));
                res.expect("steady-state large-vocab decode");
                assert_eq!(
                    bytes, 0,
                    "vocab={vocab_size} workers={workers} allocated {bytes} heap bytes"
                );
                assert_eq!(count, 0);
            });
            runs.push(logits.into_iter().map(f32::to_bits).collect());
        }
        assert_eq!(
            runs[0], runs[1],
            "vocab={vocab_size}: 1 vs 4 worker mismatch"
        );
        assert_eq!(
            runs[0], runs[2],
            "vocab={vocab_size}: 1 vs 8 worker mismatch"
        );
    }
}

fn check_split_matrix_m1_parallel_parity() {
    const ROWS: usize = 100;
    const COLS: usize = 256;
    assert_ne!(ROWS % REPACK_TILE_ROWS, 0);
    let w_bytes = encode_matrix_fast(QuantType::Q4_K_M, COLS, ROWS, 77);
    let tv = TensorView::from_bytes("split_m1", QuantType::Q4_K_M, &[COLS, ROWS], 0, &w_bytes)
        .expect("tv");
    let mat = Phi3SplitMatrix::from_tensor(&tv, ROWS, COLS).expect("split matrix");
    let x: Vec<f32> = (0..COLS).map(|c| pattern_val(1, c, 3)).collect();
    let mut seq = vec![0.0_f32; ROWS];
    mat.gemv(&mut seq, &x).expect("seq gemv");
    let seq_bits: Vec<u32> = seq.iter().copied().map(f32::to_bits).collect();

    for workers in [1_usize, 4_usize, 8_usize] {
        let pool = build_warmed_pool(workers);
        let mut par = vec![0.0_f32; ROWS];
        pool.install(|| mat.gemv_parallel(&mut par, &x))
            .expect("gemv_parallel");
        let par_bits: Vec<u32> = par.iter().copied().map(f32::to_bits).collect();
        assert_eq!(
            par_bits, seq_bits,
            "split M=1 workers={workers} bit mismatch"
        );

        #[cfg(feature = "with-kernels")]
        {
            let mut par_view = vec![0.0_f32; ROWS];
            let view_bytes = jero_kernels_rvv::QuantizedActQ8View::scratch_bytes_for_row(COLS);
            let mut act_scratch = ScratchWorkspace::new(view_bytes).expect("act view scratch");
            pool.broadcast(|_| {
                rayon::join(|| (), || ());
            });
            for _ in 0..2 {
                pool.install(|| -> Result<(), K3Error> {
                    let scope = act_scratch.scope();
                    let q_act =
                        jero_kernels_rvv::QuantizedActQ8View::quantize_row_in_scope(&x, &scope)?;
                    mat.gemv_view_parallel(&mut par_view, &q_act)?;
                    Ok(())
                })
                .expect("warmup gemv_view_parallel");
            }
            par_view.fill(0.0);
            pool.install(|| {
                let (view_res, alloc_bytes, alloc_count) = measure_allocs_in_pool(&pool, || {
                    let scope = act_scratch.scope();
                    let q_act =
                        jero_kernels_rvv::QuantizedActQ8View::quantize_row_in_scope(&x, &scope)?;
                    mat.gemv_view_parallel(&mut par_view, &q_act)
                });
                view_res.expect("gemv_view_parallel");
                assert_eq!(
                    alloc_bytes, 0,
                    "gemv_view_parallel workers={workers} allocated {alloc_bytes} bytes"
                );
                assert_eq!(alloc_count, 0);
            });
            let par_view_bits: Vec<u32> = par_view.iter().copied().map(f32::to_bits).collect();
            assert_eq!(
                par_view_bits, seq_bits,
                "split M=1 gemv_view_parallel workers={workers} bit mismatch"
            );
            assert_eq!(act_scratch.used_bytes(), 0);
        }
    }
}

fn check_counting_allocator_ignores_off_thread_noise() {
    let stop_noise = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop_noise);
    let started_clone = Arc::clone(&started);

    let noise_handle = std::thread::spawn(move || {
        started_clone.store(true, Ordering::Release);
        while !stop_clone.load(Ordering::Acquire) {
            let buf = Vec::<u8>::with_capacity(900);
            std::hint::black_box(buf.capacity());
            std::thread::yield_now();
        }
    });
    while !started.load(Ordering::Acquire) {
        std::thread::yield_now();
    }

    let (val, bytes, count) = measure_allocs(|| 42_usize);
    assert_eq!(val, 42);
    assert_eq!(
        bytes, 0,
        "off-thread noise polluted single-thread measurement"
    );
    assert_eq!(count, 0);

    let (cap, alloc_bytes, alloc_count) =
        measure_allocs(|| Vec::<u8>::with_capacity(128).capacity());
    assert!(cap >= 128);
    assert!(alloc_bytes >= 128);
    assert!(alloc_count >= 1);

    let pool = build_warmed_pool(4);
    let ((a, b), pool_bytes, pool_count) =
        measure_allocs_in_pool(&pool, || rayon::join(|| 1_usize, || 2_usize));
    assert_eq!((a, b), (1, 2));
    assert_eq!(pool_bytes, 0, "off-pool noise polluted pool measurement");
    assert_eq!(pool_count, 0);

    let ((cap_a, cap_b), pool_alloc_bytes, pool_alloc_count) =
        measure_allocs_in_pool(&pool, || {
            rayon::join(
                || Vec::<u8>::with_capacity(128).capacity(),
                || Vec::<u8>::with_capacity(256).capacity(),
            )
        });
    assert!(cap_a >= 128 && cap_b >= 256);
    assert!(pool_alloc_bytes >= 384);
    assert!(pool_alloc_count >= 2);

    stop_noise.store(true, Ordering::Release);
    noise_handle.join().expect("join noise thread");
}

/// Single sequential test runner inside this binary so `#[global_allocator]` never observes
/// `libtest` main-thread test-spawning or parallel sibling-test allocations.
#[test]
fn test_wave4_zero_alloc_decode_and_gemv_suite() {
    check_counting_allocator_ignores_off_thread_noise();
    check_layer_descriptors_and_synthetic_decode_zero_alloc();
    check_authentic_decode_zero_alloc_steady_state();
    check_deterministic_1_4_8_worker_decode_parity();
    check_200k_lm_head_parallel_decode_parity_and_zero_alloc();
    check_split_matrix_m1_parallel_parity();
}
