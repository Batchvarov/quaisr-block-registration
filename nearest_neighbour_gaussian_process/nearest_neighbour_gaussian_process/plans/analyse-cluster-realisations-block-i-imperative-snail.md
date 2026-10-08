# Cluster realisations without a hand-drawn region

## Context

The `cluster_realisations` block groups GP posterior realisations into scenarios. Today it only gives
more than one cluster when the user supplies a `mask` bounding box by hand. Without a mask it almost
always returns a single cluster. The goal is a clustering approach that needs no manual region.

Three separate causes produce that behaviour.

**1. The features cannot see magnitude.** `common/upscaling_tools/src/upscaling_tools/utils.py:77-89`
builds each feature with Canny edge detection:

```python
image = (image - image.mean()) / std
skeleton = ski.feature.canny(image, sigma=5, use_quantiles=True, low_threshold=0.2, high_threshold=0.7)
```

Line 82 divides by the standard deviation, so scaling a whole realisation changes nothing.
`use_quantiles=True` makes `0.2` and `0.7` quantiles of the gradient magnitude, and the pixel set
above a quantile does not move under a monotonic rescale of the gradient. Every realisation
therefore yields about the same number of edge pixels, by construction. A realisation at 0.5-1.5%
grade and one at 5-15% grade with the same spatial shape give near-identical feature vectors. Only
edge *position* survives, and on a smooth posterior field the gradient is shallow, so that position
jumps a long way for a small change in the field.

**2. The variance filter is a crude stand-in for the mask.** `core.py:234-239` keeps only pixels
above the 80th percentile of across-realisation variance. The threshold depends on the composition
of the whole grid.

**3. The mask does two unrelated jobs.** `core.py:178-184` restricts the clustering domain, and the
same input also drives the optional `masked` export. An empty mask extent silently forces every
realisation into one cluster (`core.py:213-219`).

The grid already carries what is needed to replace the manual region. The GP inference block writes a
`distance_mask` variable — True within 2x the median training-data spacing and inside the convex hull
(`blocks/upscaling/upscaling_gaussian_process_inference/upscaling_gaussian_process_inference/core.py:153-160`).

### Decisions taken

- Cluster on mean-centred field residuals, not on Canny skeletons.
- Report unimodality honestly. A single cluster is a valid, explained answer, decided by a gap
  statistic rather than an elbow.
- Restrict the clustering cells to `distance_mask` when the grid carries it, otherwise the full grid.
- `mask` stops affecting clustering and drives the `masked` export only.

The block type signature does not change. `__main__.py` needs no edit and no workflow breaks.

---

## Approach

### New module: `upscaling_tools/clustering.py`

Real tests in this repo live in `common/upscaling_tools/src/tests/`; the `tests/test.py` files inside
blocks are empty. Put the new logic in the shared library so it is testable by the existing
convention, then sync the wheel.

Create `common/upscaling_tools/src/upscaling_tools/clustering.py` with four functions.

**`residual_features(dataset, labels, xname, yname, cell_mask=None, max_side=256)`**

```python
X = np.stack([dataset[l].transpose(yname, xname).values.astype(np.float32) for l in labels])
valid = np.isfinite(X).all(axis=0)
if cell_mask is not None:
    valid &= cell_mask
Xf = np.nan_to_num(X)
mu = Xf.mean(axis=0)                       # pixel-wise posterior mean
R = np.where(valid, Xf - mu, 0.0)          # anomaly field per realisation
return np.stack([_box_downsample(r, max_side).ravel() for r in R])
```

Two realisations are now close when they depart from the posterior mean by a similar amount in
similar places. Cells where the realisations agree have residuals near zero for every `i`, so they
contribute nothing to any pairwise distance and mute themselves. Cells where they disagree dominate.
That is a continuous version of the region of interest, with no threshold and no user rectangle, so
the 80th-percentile filter at `core.py:234-239` is deleted rather than retuned.

Invalid cells are set to 0, which is the neutral value after centring — no separate column-dropping
step is needed.

Do not standardise per cell. Dividing by a per-cell standard deviation would amplify cells where
nothing happens.

Move the BOX downsample from `core.py:193-208` into `_box_downsample` in this module, unchanged.
The comment there explaining why BOX beats NEAREST applies to residual fields as well.

**`gap_statistic(X, k_max, n_refs=20, random_state=0)`**

Tibshirani's gap statistic. It is the standard test for "are there any clusters at all", and unlike a
silhouette score it can return k=1.

For each k, compare `log(WCSS)` of the data against the mean `log(WCSS)` of `n_refs` uniform
reference sets drawn over the per-dimension bounding box of `X`. Select the smallest k satisfying
`gap[k] >= gap[k+1] - s[k+1]`, where `s = std(ref) * sqrt(1 + 1/n_refs)`.

The caller passes `X` already in PCA coordinates, so the reference box is aligned to the principal
axes. That is Tibshirani's recommended reference distribution, so the existing PCA step earns its
place instead of being incidental.

Guard the degenerate case: if the total sum of squares is 0, every realisation is identical — return
k=1 without taking a logarithm of zero.

**`choose_clustering(X, k_max, random_state=0)`**

Runs `gap_statistic`, fits the chosen `KMeans`, and returns the fitted model plus a small report
(chosen k, the gap values, and `sklearn.metrics.silhouette_score` when k > 1). The silhouette is
reported for the user's judgement only; it is not a selection criterion, because it cannot express
k=1.

Pass `random_state` to every `KMeans`. The current code at `core.py:247` does not, so the same input
can produce different clusters on different runs.

**`k_max_for(n_labels)`** — returns `min(9, n_labels - 1)`.

`core.py:246` uses `range(1, min(10, len(labels) - 1))`, which caps k at `min(9, N - 2)` and fits only
k=1 when N=3, despite the README asking for at least 3 realisations.

### Rewrite the clustering section of `core.py`

File: `blocks/upscaling/cluster_realisations/cluster_realisations/core.py`, lines 177-271.

1. Keep `make_skeletons` (`core.py:153-157`). Skeletons stay, but only as the `features` GeoTIFF
   viz layer. They leave the clustering path.
2. Pick the clustering cells:

   ```python
   cell_mask = None
   if "distance_mask" in dataset:
       m = dataset["distance_mask"].transpose(yname, xname).values.astype(bool)
       if m.any():
           cell_mask = m
   ```

   An absent or all-False `distance_mask` falls back to the full grid. Do not add a minimum-cell
   threshold: a small mask is a genuine region of interest, and a threshold is the kind of tuned
   number this change removes.
3. Build features with `residual_features`, keep `PCA(n_components=0.95)` at `core.py:242`, then call
   `choose_clustering`.
4. Delete the `empty_extent` branch (`core.py:210-219`). Clustering no longer depends on `mask`, so an
   empty mask extent must not collapse the clusters. Instead, when the mask extent is empty, skip the
   `masked` export and say so in `log`.
5. Confine the `mask` handling at `core.py:178-184` to the export path. The uncommitted
   `export_masked = mask is not None` change already routes `export_specs` correctly; keep it.

### `log` output

Report the chosen k, and when k=1 state plainly what that means:

> The realisations form one continuous family — the gap statistic found no separated groups. This is
> the expected result for posterior samples of a unimodal field.

A user who reads bare "1 cluster" will read it as a failure. When k > 1, report k and the silhouette
score. When the number of realisations is small the test has little power; say so.

---

## Files to change

| File | Change |
| :--- | :--- |
| `common/upscaling_tools/src/upscaling_tools/clustering.py` | New. `residual_features`, `gap_statistic`, `choose_clustering`, `k_max_for`, `_box_downsample`. |
| `common/upscaling_tools/src/tests/test_clustering.py` | New. See verification. |
| `common/upscaling_tools/pyproject.toml` | Bump version from `0.3.11`. |
| `blocks/upscaling/cluster_realisations/cluster_realisations/core.py` | Rewrite lines 177-271. Delete the variance filter and the `empty_extent` branch. Confine `mask` to the export path. |
| `blocks/upscaling/cluster_realisations/README.md` | Rewrite the algorithm section, the `mask` row, and the `log` row. Delete the "Why the pixel-variance filter" section. |
| Block manifests | Run `scripts/sync_tools.sh` to rebuild and repin the wheel, as commit `48e280f` did. |

`common/upscaling_tools/src/upscaling_tools/utils.py` is not touched. `make_skeletons` stays as it is,
because the `features` export still uses it.

`__main__.py` is not touched. No input or output type changes.

---

## Verification

**Unit tests** — `common/upscaling_tools/src/tests/test_clustering.py`, run with
`pytest common/upscaling_tools/src/tests/test_clustering.py` from the repo root. Follow the plain
`import pytest` + numpy style of `test_spectral.py`.

1. **Magnitude is visible.** Build realisations with identical spatial shape but different
   amplitudes. Assert `choose_clustering` separates them. This is the regression test for the Canny
   blindness — build the same fields through `make_skeletons` and assert those features do *not*
   separate them, so the test records why the feature changed.
2. **Two real groups are found.** Two well-separated families of fields plus noise. Assert k=2 and a
   correct assignment up to label permutation.
3. **One cloud stays one cluster.** Independent Gaussian fields with no group structure. Assert k=1.
4. **`distance_mask` restricts the cells.** Put a strong signal outside the mask and none inside.
   Assert the features built with `cell_mask` are near zero, and that the same data without a mask
   produces non-zero features.
5. **NaN cells are safe.** NaN in a subset of realisations. Assert the feature matrix is finite and
   the NaN cells contribute 0.
6. **Determinism.** Two calls with the same `random_state` give identical labels.

**End-to-end** — run the block on a real GP inference output with no `mask`, and confirm:

- More than one cluster appears where the user previously needed a hand-drawn region, or a k=1 result
  arrives with the explanatory `log` message.
- `cluster_zip` holds `images` and `features` per cluster, and no `masked` directory.
- Re-run with a `mask` and confirm the `masked` export reappears and the cluster labels are unchanged
  from the run without it, which is the point of making `mask` export-only.

---

## Risks

- **The honest answer is often k=1.** Posterior samples of a unimodal field genuinely form one
  continuous cloud. With this option the block will say so instead of manufacturing groups. If that
  turns out to be unhelpful in practice, the fallback is the "always k scenarios" design: choose a
  fixed k and report the silhouette alongside it. That is a later, separate change.
- **Few realisations means low power.** With N of about 3-10, PCA retains close to N-1 components and
  pairwise distances concentrate, so the gap statistic will rarely reject k=1. The `log` must state
  this rather than leave the user guessing.
- **Cluster meaning changes.** Clusters group by where and how much grade departs from the posterior
  mean, not by edge geometry. Results on existing workflows will differ. This is the intended effect,
  but it is a visible change for anyone comparing against an earlier run.
