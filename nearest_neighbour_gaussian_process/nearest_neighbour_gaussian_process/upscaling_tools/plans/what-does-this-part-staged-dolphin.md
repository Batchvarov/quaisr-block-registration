# Finish the gravity-upscaling GPU OOM fix

## Context

`demo_blocks/gravity_upscaling/upscaling_gaussian_process_inference` dies during
`model.sample_gp` with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.76 GiB.
GPU 0 has a total capacity of 14.74 GiB of which 410.19 MiB is free.
Process 15237 has 14.34 GiB memory in use. Of the allocated memory 11.63 GiB is
allocated by PyTorch, and 1.04 GiB is reserved by PyTorch but unallocated.
```

Single process, so the whole 14.34 GiB is the block's own. 12.67 GiB of it is the
PyTorch caching allocator; the remaining ~1.67 GiB is CUDA context + cuBLAS/cuFFT
workspaces and plan caches, which `empty_cache()` cannot reclaim.

The pathwise sampler (`sample_gp` → `sample_pathwise` → the FFT prior) held several
arrays that each scale with the full output grid `nx*ny` and with `n_samples`. With
`padding_factor=2` the padded lag grid is `4*nx*ny` nodes, and the failing 1.76 GiB
request equals `32*nx*ny` bytes — the complex64 buffer behind `torch.fft.fftn(c).real`
at roughly `nx*ny ≈ 59M` cells.

**Already done** (in `packages/upscaling_tools`, verified, not yet shipped):

- `nngp_fft_prior.py::_kernel_first_row` — enumerate the padded lag grid in flat
  chunks instead of `meshgrid` + `stack`. Removes ~`80*nx*ny` bytes of transients.
- `nngp_fft_prior.py::_fft_sample_stationary` — circulant draw rewritten from
  `fftn`/`ifftn` on complex noise to `rfftn`/`irfftn` on real noise. The old form
  drew complex noise and discarded `z.imag`; the real form is distributionally
  identical at half the memory. Spectrum now stored half-size. Per-chunk peak
  32 → ~12 bytes/elem.
- `DEFAULT_FFT_PEAK_BUDGET_BYTES` (512 MiB) replaces the hardcoded 1 GiB chunk
  budget, plumbed through `GridFFTPath` / `GridFFTPriorSampler` and
  `nngp_sampling.py::_draw_matheron_paths_NNGPModel_fallback` as
  `fft_peak_budget_bytes`.
- `nngp.py::sample_pathwise` — new `sample_batch_size` caps samples per Matheron
  path, so `GridFFTPath._draws` scales with the batch rather than `n_samples`; each
  evaluated chunk is moved to the host immediately so the `(S, M)` output never
  accumulates on the device.

Verification so far: `src/tests/test_fft_prior.py` 8/8 pass; chunked lag grid matches
the `meshgrid` version to 1e-7 in 2D and 3D at even and odd padding; empirical
covariance vs analytic RBF is 0.04 relative (2D) and 0.08 (3D) at S=4000; a forced
`chunk=1` path works.

**Outcome wanted:** the block runs to completion on a 14.74 GiB GPU, with the memory
knobs reachable from the workflow rather than only from Python defaults.

## Remaining work

### 1. Expose the new knobs on the block

`demo_blocks/gravity_upscaling/upscaling_gaussian_process_inference/upscaling_gaussian_process_inference/core.py`

Add to `InferenceOptions` alongside the existing chunk-size fields:

- `sample_batch_size: Optional[int] = 2` — samples per Matheron path. This is the
  single biggest lever: `_draws` and the per-path weight both scale with it.
- `fft_peak_budget_mb: Optional[int] = 512` — converted to bytes and passed as
  `fft_peak_budget_bytes`.

Both go into `sample_kwargs` in the `use_fft_prior` branch (`fft_peak_budget_bytes`
only there; `sample_batch_size` applies to both prior paths). Follow the existing
`x = inference_options.y or default` idiom already used for the other four options.

`.../upscaling_gaussian_process_inference/__main__.py`

Add the two fields to `inference_options_type` as `QuaisrOptionalType(INTEGER)`.
Note `num_random_features` and `use_fft_prior` / `fft_padding_factor` are already
out of sync between `InferenceOptions` and the declared record type — add the new
fields and bring the record type back in line with the dataclass while there.

### 2. Cap the cuFFT plan cache

The ~1.67 GiB of non-allocator overhead is partly cuFFT plan work areas, one per
distinct transform shape. In `core.py`, before sampling:

```python
if torch.cuda.is_available():
    torch.backends.cuda.cufft_plan_cache.max_size = 1
```

The sampler now issues at most two distinct shapes (the padded forward and inverse
transforms), so a cache of 1 costs little and bounds the hidden footprint.

### 3. Ship the wheel

`packages/upscaling_tools/pyproject.toml` is at `0.3.3`; the block pins
`upscaling_tools-0.3.3-py3-none-any.whl`. Bump to `0.3.5` (0.3.4 wheels already exist
in `dist/`), build, then sync:

```bash
cd packages/upscaling_tools && uv build --out-dir ../../dist
cd ../.. && ./copy_tools.sh
```

`copy_tools.sh` rewrites only the `botorch` path in each block `pyproject.toml`, so
the `upscaling-tools` wheel path in the block's `[tool.uv.sources]` must be updated
by hand to the new version, and `uv.lock` regenerated.

## Verification

1. `cd packages/upscaling_tools && uv run --frozen --with pytest python -m pytest src/tests/ -q`
   — `test_fft_prior.py` and `test_hierarchical_vecchia.py` both green.
2. Statistical guard, run on CPU: draw from `_fft_sample_stationary` at
   `sample_batch_size` 1 and at `n_samples`, confirm the empirical covariance matches
   the analytic kernel in both cases — batching must not change the distribution.
3. On the GPU box, rerun the block on the same inputs that produced the OOM. Wrap the
   sampling call with `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`
   and print both, so the headroom is a number rather than "it didn't crash".
4. Compare `results/gp_inference_spacing=*.nc` against a pre-change run at a small
   grid spacing: `GP_mean_pred` and `GP_variance_pred` must match bit-for-bit (the
   inference path is untouched); the `GP_sample_*_pred` fields will differ because the
   noise draw changed, so check their mean and variance across samples instead.

## Open question

If it still OOMs after this, the next lever is tiling the output grid — building one
`GridFFTPath` per tile instead of one over the whole bounding box. That removes the
last `O(nx*ny)` resident buffers but changes the prior draw at tile seams, so it needs
a design decision before being attempted.
