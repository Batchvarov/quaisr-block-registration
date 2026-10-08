# Developer notes: auto-kernel parameter choices

Why the auto-constructed kernel in `construct_multiscale_kernel` has the bounds
it has. Read this before changing any of them.

Maintainer notes, not user documentation. The block README describes what the
block does and how to configure it; this file records the evidence behind the
hardcoded values and the dead ends already ruled out.

All measurements below come from one dataset: a 2586 × 3925 ERS raster at 250 m
cells, 57% valid, values of mean −7.4 and standard deviation 84. It was
decimated by 10 to a 259 × 393 grid, trained on 20,000 points with
`ninducing` 12,000 (7,689 realised) for 250 epochs, and sampled with 4
realisations. See [Caveats](#caveats).

## Read this first: the caps are calibrated outside the production regime

Both outputscale caps were fitted at `ninducing` 12,000. Production runs
800,000. Because `min_lengthscale` is a multiple of the realised inducing pitch,
that changes which feature sizes the kernel can represent by almost an order of
magnitude:

| `ninducing` | grid | pitch | short scale | mid scale | long scale |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 12,000 (these measurements) | 135×89 | 7.3 km | 14.5 km | 65.3 km | 293.7 km |
| 800,000 (production) | 1102×726 | 0.89 km | 1.78 km | 22.9 km | 293.7 km |

Every figure in this file was measured with `MIN_LENGTHSCALE_SPACINGS = 2.0`,
including the two rows above. The constant is now 0.25, which divides both short
scales by eight and both mid scales by roughly three. The measurements are left
as they were taken; see
[`MIN_LENGTHSCALE_SPACINGS` and the lengthscale starts](#min_lengthscale_spacings-and-the-lengthscale-starts)
for what changed and why.

The example raster's own correlation falls to 0.5 by 10.1 km and to 0.1 by
31.6 km. At `ninducing` 12,000 the shortest component the kernel permits is
14.5 km — past the field's half-correlation length, so the model could not
represent the dominant structure at all. At 800,000 the short and mid components
land at 1.78 km and 22.9 km, which brackets it.

Two consequences for everything below:

- The cap values were chosen where the model was structurally blind to the data.
  They may not transfer.
- At production scales the variance budget looks inverted: the short component
  is capped at 0.1 where the measured correlation puts 0.243, while the long
  component, at a scale where the field has no structure, is allowed 2.0 where
  the measured correlation puts 0.024.

Treat `MAX_OUTPUTSCALE = 2.0` and `MAX_ROUGH_OUTPUTSCALE = 0.1` as bounds known
to prevent a catastrophic failure, not as values known to be right.

## Summary

| Parameter | Value | Set by | Confidence |
| :--- | :--- | :--- | :--- |
| `MAX_OUTPUTSCALE` | 2.0 | sampled realisations, not the training loss | calibrated at `ninducing` 12,000 only |
| `MAX_ROUGH_OUTPUTSCALE` | 0.1 | inducing spacing — resolvability | value not established; see below |
| `MIN_OUTPUTSCALE` | 1e-3 | avoiding a negative outputscale | purpose sound, magnitude arbitrary |
| `MIN_LENGTHSCALE_SPACINGS` | 0.25 | held-out calibration and realisation texture | one survey, three repeats |
| `long_lengthscale` | $d_{\text{bbox}}/4$ | survey outline | 2.8x the scale the measured correlation puts it at |
| lengthscale start | $1.5 \times$ bound | gpytorch's default start value | sound |

Every bound that can bind, binds. Across every density and cap setting tested,
the short component sits on its lengthscale floor *and* its outputscale ceiling,
the mid component on both of its, and the long component on its outputscale
floor with its lengthscale fled past the domain. No component finds an interior
optimum, so these are effectively hardcoded hyperparameters rather than a prior
the data speaks through.

## `MAX_OUTPUTSCALE` — the upper bound on the long and mid components

### The failure it prevents

Left uncapped, the mid-scale component's outputscale trains to about 99 against
outputs standardised to unit variance. The sampled realisations then run 14
standard deviations past the observed data range — a field whose data spans
−1561 to 1318 produces samples from −2117 to 2047, with gap variance nearly 5×
the variance over covered ground.

This was the defect behind the extreme values reported in cluster realisation
rasters, and it is older than the kernel shape that made it visible. Measured at
`ninducing` 12,000, trained prior variance by kernel revision:

| Kernel | Trained prior variance | Gap ratio $r$ | Samples vs observed range |
| :--- | ---: | ---: | :--- |
| current, capped | 2.10 | 1.09 | inside |
| uncapped, short mid component | 99.24 | 4.92 | +14.3 sd |
| uncapped, long mid component | 58.40 | 1.52 | inside |

The third row is the kernel that shipped before the one that produced the
reported rasters. Its outputscale also ran away — variance 58 against a signal
standardised to 1 — but its uncapped component sat at a long lengthscale, so the
excess appeared as smooth regional bias rather than blobs and went unnoticed.
Moving that component to a shorter lengthscale did not create the runaway; it
made the same runaway legible.

Two lessons. The cap is not a fix for a recent regression, it closes a defect
every revision of this kernel has carried. And the runaway strengthens with
inducing density — an earlier reading of 2.47 for the third row came from
`ninducing` 3,000 — so a configuration that looks safe at test scale may not be
at production scale.

### Why it cannot be a slack bound

The variational ELBO is monotone in the outputscale of whichever component
carries the signal, so the fit pins that component on whatever ceiling it is
given. Measured at caps of 0.4, 1, 2, 4, 8, 16, 32 and 64, the component sat on
the cap every time, to five decimal places. There is no cap value loose enough
for the fit to find an interior optimum, so **the cap is the effective value of
that parameter on every dataset**, not a limit the fit normally stays clear of.

Two consequences:

- Changing `MAX_OUTPUTSCALE` changes the model on every dataset, not only on
  ones that would otherwise blow up.
- The components that are *not* carrying the signal pin on `MIN_OUTPUTSCALE`
  instead. Both bounds bind; nothing sits in between.

### Why the training loss cannot choose it

The loss improves monotonically as the samples get worse:

| Cap | Training loss | Gap variance ratio $r$ | Samples beyond data range |
| ---: | ---: | ---: | ---: |
| 0.4 | 3719 | 0.80 | inside |
| 2 | 306 | 1.15 | inside |
| 4 | 718 | 1.27 | inside |
| 8 | 232 | 1.76 | inside |
| 16 | 57 | 2.4 | +5.8 sd |
| 32 | — | 2.9 | +2.6 sd |
| 64 | — | 4.2 | +11.1 sd |

$r$ is the ratio of sample standard deviation in the uncovered gaps to that over
covered ground; $r \approx 1$ means the gaps are as variable as the observed
ground. Neither the loss nor any metric on the posterior mean detects the
failure — only the sample statistics do. This is why the defect reached
production: the training diagnostics looked better as it got worse.

### Where 2.0 comes from

From $r$, which rises smoothly with the cap and offers no threshold to pick. 2.0
is the value that puts $r$ closest to parity while staying far below the runaway.
The committed configuration measures $r = 1.09$ with samples inside the observed
range.

### What it does not rest on

The fixed likelihood noise. `construct_likelihood` pins the noise and sets
`requires_grad_(False)`, and a plausible story is that the clamped noise forces
an irreducible misfit into the outputscale instead. That story is wrong: raising
the fixed noise from $10^{-4}$ to $0.3$ moves the pinned outputscale in the
fifth decimal place. It does drop the loss from 1812 to 2.0, so the noise does
absorb the misfit — the outputscale climbs to the ceiling anyway.

### What is still unexplained

Why the ELBO is monotone in the outputscale at all. One untested candidate: with
a mean-field `q(u)`, a wider prior always reduces `KL[q(u) || p(u)]`, so
inflating the prior buys ELBO regardless of whether it helps prediction. That
would be noise-independent, which matches the observation. **The cap bounds this
pathology; it does not fix it.**

## `MAX_ROUGH_OUTPUTSCALE` — the tighter bound on the short component

**The principle is sound; the value 0.1 is not established.** Treat this section
as the weakest in this file.

The principle: the model cannot resolve structure finer than the inducing
spacing, so variance placed on a component below that spacing becomes
unresolvable white noise rather than signal, and a fit that shrinks into it
produces a speckled field.

What was actually measured, at `ninducing` 12,000: raising this from 0.1 to 2.0
moved the short component's outputscale to 1.99 and collapsed the mid one to
0.02, and the loss got worse (1194 against 164).

Three reasons not to lean on that:

- **The samples from that run were never inspected.** The claim that the fit
  "turns to speckle" was inferred from the loss and from a pre-fix raster
  produced by a different configuration, not from the realisations of that run.
- **The short component was at 14.5 km in that run** — nowhere near speckle
  scale, and comfortably resolved by the 2.5 km evaluation grid. The field's
  measured correlation is 0.33 at 14.5 km and about 0.015 at the mid component's
  65 km, so the fit preferring the short component may have been correct.
- **At production scales the short component is 1.78 km**, which is resolvable
  and inside the band the field varies over. Fitted to the measured correlation
  at that floor, the ladder wants 0.243 of the variance on a 3.5 km component,
  which a 0.1 cap forbids; it is forced onto the mid component instead. That
  would produce fields too smooth at short range and too energetic at 20–60 km.

So this bound is doing something real — an unbounded short component is a known
failure mode — but 0.1 was neither derived nor validated at the scale production
runs at. Re-measure before trusting it.

## `MIN_OUTPUTSCALE` — the floor

`lessthan` in `parse_constraint` maps to gpytorch's `LessThan`, whose lower
bound is $-\infty$: a raw value of $-20$ gives an outputscale of $-19.9$, and a
negative outputscale makes the kernel matrix indefinite, failing the first
variational KL Cholesky. Every outputscale therefore uses `interval`, which
bounds both sides. `interval` also needs no explicit initial value, because
gpytorch maps a raw of 0 to the interval midpoint.

The floor is an absolute $10^{-3}$ rather than a fraction of the cap. A relative
floor of `cap/100` moved the non-signal components whenever the cap changed —
they pin on the floor, so they sat at 0.04, 0.08 and 0.16 at caps of 4, 8 and 16.

## `MIN_LENGTHSCALE_SPACINGS` and the lengthscale starts

`min_lengthscale = MIN_LENGTHSCALE_SPACINGS × s`, where $s$ is the realised
median inducing spacing.

**This was 2.0 and is now 0.25.** The old value rested on the argument that the
model cannot resolve structure below the inducing spacing, so a shorter bound
only gives the fit a white-noise component to shrink into. The first half is
true and the conclusion does not follow: the component is drawn from the prior
below the spacing rather than inferred, which is what a realisation should do
with unresolved structure, and the variance has to go somewhere. Left out of the
kernel it reappears as held-out error the model does not report.

Measured on `VAL_GRAV_VARIOUS.csv` at `ninducing` 2,000 (1,317 realised, spacing
1.96 km), three runs per setting, held-out on 790 points:

| spacings | floor | `error_rate` | `predictive_nugget` | rmse |
| ---: | ---: | :--- | :--- | :--- |
| 2.0 | 3.91 km | 0.349 / 0.361 / 0.604 | 0.303 / 0.352 / 4.978 | 0.607 / 0.617 / 2.085 |
| 1.0 | 1.96 km | 0.185 / 0.187 / 0.261 | 0.324 / 0.334 / 0.675 | 0.631 / 0.635 / 0.864 |
| 0.5 | 0.98 km | 0.077 / 0.081 / 0.097 | 0.107 / 0.121 / 0.201 | 0.602 / 0.608 / 0.684 |
| 0.25 | 0.49 km | 0.058 / 0.059 / 0.066 | 0.037 / 0.041 / 0.102 | 0.614 / 0.630 / 0.634 |
| 0.125 | 0.24 km | 0.056 / 0.057 / 0.062 | 0.022 / 0.023 / 0.062 | 0.650 / 0.650 / 0.695 |

Nominal `error_rate` is 0.05. 0.25 is the knee: 0.125 barely improves
calibration and starts costing rmse. The 2.0 row also threw the only
catastrophic run in the whole sweep; no lowered floor produced one.

### It frees the mid component, which is the larger effect

`mid = sqrt(long × min)`, so the floor moves two components. At 2.0 the mid
floor is 10.98 km and the fit sits on it; at 0.25 the floor is 3.88 km and the
fit finds an interior optimum near 6.5 km. Two runs per setting:

| spacings | mid lengthscale | mid outputscale | short outputscale | prior variance |
| ---: | :--- | :--- | :--- | :--- |
| 2.0 | 11.45 / 11.45 km | 1.454 / 1.520 | 0.0956 / 0.0951 (on the cap) | 1.643 / 1.714 |
| 0.5 | 7.50 / 7.49 km | 0.530 / 0.510 | 0.0795 / 0.0803 | 0.697 / 0.665 |
| 0.25 | 6.66 / 6.32 km | 0.516 / 0.471 | 0.0648 / 0.0653 | 0.660 / 0.604 |

At 2.0 the mid component alone carries more variance than the whole
standardised signal, which is the outputscale inflation this file attributes to
the ELBO. Lowering the floor cuts total prior variance by about 60% and takes
the short component off `MAX_ROUGH_OUTPUTSCALE` as well. The **long**
lengthscale becomes irreproducible in exchange (53.5 against 108.5 km between
repeats at 0.25, against 91.8/93.5 at 2.0); it holds about 12% of the prior
variance, so it moves the fit little.

### Realisation texture

Roughness here is the sd of the one-cell difference over the field sd, on a
247 m grid. The field's own value follows from the measured `nn_variogram` of
0.0407 standardised at a 398 m point spacing, giving about 0.28 — an upper
bound, since part of that is measurement noise rather than signal.

| spacings | realisation roughness | posterior mean roughness |
| ---: | ---: | ---: |
| 2.0 | 0.115 | 0.098 |
| 1.0 | 0.163 | 0.104 |
| 0.5 | 0.209 | 0.092 |
| 0.25 | 0.250 | 0.085 |

The shipped realisations were less than half as rough as the field. The
posterior mean is **not** affected — it is a conditional expectation, so the
extra short-scale prior variance averages out of it, and by this measure the
lowered floor gives the smoothest mean of the five. Local blobs in the mean at
sparse training points are not caused by the floor either: the median excursion
at 309 isolated points is 0.0517 field sd at 2.0 against 0.0385 at 0.25.

### The bound cannot simply be removed

`lengthscale_at` starts the component at 1.5x the bound, so the floor is also
the initialisation, and that is the half that matters. Removing the bound and
starting the short component where the shipped kernel starts it leaves it at
1.15-1.24 km, which is how far it descended rather than an optimum it found.
Removing the bound and starting at the bound leaves it at zero lengthscale
acting as a nugget: rmse 0.794 / 0.800 / 3.672, with one run in three diverging.
Capping the outputscale does not substitute for the floor -- the cap bounds how
much variance the component carries, the floor decides at what wavelength.

A lengthscale travels only a bounded distance per training run, so these bounds
place the ladder rather than merely constraining it. That is the same mechanism
as "every bound that can bind, binds" in the [Summary](#summary).

### What this rests on

One survey, three runs per setting, at `ninducing` 2,000 against production's
800,000. Run-to-run spread at a fixed setting reaches 0.15 in rmse, so the
separation between 0.5, 0.25 and 0.125 is inside the noise; the separation from
2.0 is not.

### The lengthscale starts

Each lengthscale *starts* at $1.5\times$ its bound. A gpytorch constraint
sets a bound, not a start value: `raw_lengthscale` initialises to zero, so a
bounded lengthscale starts at `bound + softplus(0)`, about `bound + 0.69`. That
offset swamps any bound well below it, so without an explicit initial value the
short and mid components both start near 0.69 and model the same wavelength
however far apart their bounds are. The $1.5$ factor is headroom — an initial
value sitting exactly on the bound inverts to `raw = -inf`, which would then be
written into the saved state dict.

## Vecchia DAG

Not a kernel parameter, but the same kind of choice: `use_hierarchical_vecchia`
decides which correlations the NNGP prior can hold at all, so it interacts with
the lengthscale bounds above.

A flat DAG gives each inducing point the `k` nearest points preceding it in
Hilbert order. Those all sit in one small disc, so the prior treats distant
regions as close to independent and long-range structure is lost.

The hierarchical DAG spends the same budget of `k` across scales. It splits the
inducing points into levels: `L_0` is a global skeleton of `k` points spread
over the domain, and the rest are divided into geometrically growing levels.
Each point takes some parents from `L_0` (long range), some from intermediate
levels (medium range), and the rest from earlier points in its own level
(local). The cost stays $O(Mk^2)$.

Prefer the hierarchical DAG when the correlation range covers several inducing
spacings. For data whose range is about one spacing there is no long-range
structure to recover, the distant parents carry no information, and the flat DAG
is the better approximation.

Both settings were measured against the outputscale runaway and neither prevents
it: the choice of DAG changes which correlations are representable, not whether
the outputscale pins.

## Reopened: deriving the scale ladder from the data

`long_lengthscale = bbox_diagonal / 4` is a property of the survey outline, not
of the field. On this raster it lands at 293.5 km, against a measured
correlation that reaches 0.1 by 31.6 km. The ladder was rewritten once to take
the long scale from an empirical correlation range estimated off the training
points, measured 33.1 km, and was **rejected on the wrong criterion**. Read this
section before repeating either the derivation or the rejection.

### What the rejection rested on

Trained head to head at `ninducing` 100,000 on the same data for the same 250
epochs, the data-derived ladder fit worse: ELBO loss 357.0 against 270.3, gap
variance ratio 1.31 against 1.13, identical prior variance to four decimals. A
sweep of the mid floor over 10, 15, 22 and 38 km moved held-out RMSE only from
30.7 to 29.5 against a field standard deviation of 84, every cell at
$R^2 \approx 0.88$, while calibration degraded — the fraction of held-out points
outside the 95% interval rose from 6.3% to 8.6% and the regression slope fell
from 0.919 to 0.900.

None of ELBO loss, gap variance ratio or held-out error measures whether the
trained kernel's correlation function matches the field's, which is what the
ladder sets. The ELBO is worse than uninformative here: it is monotone in the
outputscale of whichever component carries the signal (see
[Why the training loss cannot choose it](#why-the-training-loss-cannot-choose-it)),
so it rewards exactly the runaway the caps exist to stop.

### What survives

The **long** component flees whatever floor it is given. From a 293 km floor it
trains to 2,785 km; from a 33 km floor it trains to 3,311 km. Its outputscale
pins at `MIN_OUTPUTSCALE` both times. Deriving the long scale from the data does
not move the trained model, so config B below is expected to behave much like
config A.

### What points the other way

The **mid** component pins on its floor and carries the field. At `ninducing`
12,000 its floor is 65.3 km and it trains to 65 km holding 95.2% of the prior
variance. The floor value *is* the model's dominant wavelength, so a floor
derived from the measured correlation sets the model's correlation directly.

Read that together with "every bound that can bind, binds" in the
[Summary](#summary). Under a wavelength criterion that is the mechanism, not the
defect: if the fit sits on its bounds, bounds taken from the data's correlation
function put the model's correlation where the data's is.

### What the field actually looks like

Measured with `upscaling_tools.spectral.empirical_correlation` on 150,000 points
drawn from the valid cells of the reference raster, 4,000,000 pairs over 50 log
spaced bins, with a linear trend removed:

| Lag | Measured correlation |
| ---: | ---: |
| 1 km | 0.98 |
| 2 km | 0.83 |
| 5 km | 0.69 |
| 10 km | 0.50 |
| 15 km | 0.32 |
| 26 km | 0.14 |
| 50 km | 0.04 |
| 100 km | 0.00 |

Correlation 0.5 at 10.1 km, 0.1 at 31.6 km, 0.05 at 40.5 km.

Fitting an additive Matern sum to that curve with
`upscaling_tools.spectral.fit_ladder`, floored at the resolvable scale for each
inducing density:

| Short floor | Fitted ladder | Fitted variances | Unrepresentable |
| :--- | :--- | :--- | ---: |
| 1.78 km (`ninducing` 800,000) | 105.2 / 13.9 / 3.5 km | 0.024 / 0.693 / 0.243 | 0.041 |
| 14.5 km (`ninducing` 12,000) | all three at the 14.5 km floor | 0 / 0.866 / 0 | 0.134 |

At the production floor the fit wants 0.243 of the variance on a 3.5 km
component, which `MAX_ROUGH_OUTPUTSCALE = 0.1` forbids, and only 0.024 on a
long component, which the shipped ladder places at 293.7 km and allows 2.0. At
the 12,000 floor every component collapses onto the floor, which is the
structural blindness the [regime warning](#read-this-first-the-caps-are-calibrated-outside-the-production-regime)
describes.

### This supersedes an earlier autocorrelation table

An earlier revision of this file recorded the raster's radially averaged
autocorrelation as 0.41 at 1 km, 0.23 at 10 km and 0.01 at 50 km, and concluded
from it that roughly 59% of the field's variance sits below 1.4 km. Those
numbers are not reproducible and the conclusion drawn from them is withdrawn.
Two independent estimators agree against them:

| Lag | Superseded table | `empirical_correlation` | Radial FFT, mask corrected |
| ---: | ---: | ---: | ---: |
| 1 km | 0.41 | 0.98 | 0.94 |
| 10 km | 0.23 | 0.50 | 0.50 |
| 25 km | 0.07 | 0.14 | 0.15 |
| 50 km | 0.01 | 0.04 | 0.04 |

The FFT estimator divides the autocorrelation of the zero-filled field by the
autocorrelation of the validity mask, so the 43% invalid cells do not count as
signal. Skipping that correction was the obvious suspect and is not the cause:
the uncorrected curve is within 0.02 of the corrected one at every lag above
1 km. Whatever produced the superseded table, it reported about half the
correlation at every lag.

The practical difference: the field has almost no variance below 1 km, not most
of it. The model's shortfall is at 10–30 km, not below 1.4 km.

### How to settle it

`docs/harness.py` trains one model per ladder and scores each by
`spectral_misfit`, the mean absolute difference between the trained kernel's own
correlation function and the measured one, integrated in log lag. It reports
ELBO loss and held-out error alongside, which are recorded and do not decide.

| Config | Ladder | Outputscale bounds |
| :--- | :--- | :--- |
| A | the shipped geometric ladder | the shipped caps |
| B | long scale from the measured 0.1 range | the shipped caps |
| C | `fit_ladder` lengthscales, as `interval` bounds | `fit_ladder` variances, as `interval` bounds |

Config C uses intervals rather than floors on both. Every floor in the shipped
kernel either binds or is fled, so an interval is what holds a component at the
wavelength the data puts it at.

Run the matrix at `ninducing` 100,000 first, then repeat A and C at production's
800,000. The short floor moves by almost an order of magnitude between them, so
a result at one does not transfer to the other.

### How wide C's bounds should be

Measured at `ninducing` 12,000 on the reference raster, one run per setting.

| Setting | Spectral misfit | Model 0.5 | Model 0.1 |
| :--- | ---: | ---: | ---: |
| A | 0.277 | 60.8 km | 147.3 km |
| C, fixed band 1.5 | 0.024 | 15.6 km | 34.1 km |
| C, fixed band 3 | 0.023 | 15.6 km | 34.4 km |
| C, fixed band 6 | 0.022 | 15.6 km | 33.4 km |

The field reaches 0.5 at 15.6 km and 0.1 at 32.3 km on this curve. Widening past
3 buys almost nothing: the short component sits on the resolvable floor at every
setting, so its extra freedom is unusable, and only the mid and long components
can move.

**Lengthscale bands can come from the data.** `bootstrap_ladder` resamples
spatial blocks and returns one spread per component. On this raster at
$1\sigma$ the bands are 2.71, 1.39 and 1.15 for long, mid and short. One band
for all three is therefore wrong. Which component is tightest is a property of
the data, not a rule -- on synthetic two-scale fields the order flips.

**The variance split cannot, and must still be constrained.** Across bootstrap
replicates the per-component variances vary by orders of magnitude while the
total holds, because Matern components at adjacent lengthscales are close to
collinear. Bounding only the total and letting the fit choose the split was
tried and is worse: misfit 0.095 against 0.022, with half the variance parked on
a 40 km component. The ELBO is monotone in the carrying component's outputscale,
so a free split is decided by the objective this whole section exists to
distrust. Constrain the split to the fitted one. It works because it overrides
the ELBO, not because the data pins it.

Held-out error and ELBO loss do not separate any of these settings. Two runs of
the identical configuration differ by 1.5 in RMSE and 230 in loss, which is
larger than every difference between settings. The misfit differences are
larger than that noise; the prediction differences are not.

## Rejected: a domain-scale component

A revision of this kernel carried a fourth component bounded below by the full
bounding-box diagonal, on the argument that its domain-wide correlation acted as
a random intercept tying every point to every other, and that this was what kept
gaps near the observed level.

That argument held only at initialisation. With the mid-scale cap in place the
component trained to `MIN_OUTPUTSCALE` and contributed under 1% of the prior
variance. Dropping it moved the total prior variance 2.14 → 2.12 and the sample
maximum 552 → 554, and cut 14% off the training time. It also raised the
circulant embedding's negative-eigenvalue fraction in the FFT prior sampler,
since a near-constant component wraps badly under a padding factor of 2.

The cap bounds the samples. The component did not.

## Feature sizes, in metres

The kernel works in scaled units. `GlobalMinMaxScaler` takes one global min and
max across both mean-centred columns, so the mapping is isotropic and the larger
extent sets it: for this raster, **1 scaled unit = 490,500 m**, from the 981 km
northing extent.

Trained kernel at `ninducing` 12,000, against the measured correlation of the
field. The measured column supersedes an earlier autocorrelation table; see
[This supersedes an earlier autocorrelation table](#this-supersedes-an-earlier-autocorrelation-table).

| Lag | Measured | Model correlation |
| ---: | ---: | ---: |
| 1 km | 0.98 | 1.00 |
| 5 km | 0.69 | 0.99 |
| 10 km | 0.50 | 0.96 |
| 25 km | 0.14 | 0.83 |
| 50 km | 0.04 | 0.59 |
| 100 km | 0.00 | 0.25 |

The two do not overlap. The field reaches correlation 0.5 at 10.1 km and 0.1 at
31.6 km; the model reaches 0.5 only at 60 km and 0.1 at 145 km. Per component:
the mid term held 95.2% of the prior variance at a 65 km lengthscale, the short
term 4.8% at 14.6 km, the long term 0.1% at 318 km. The model is too smooth
across the whole band the field varies over.

The resolvable floor is a configuration choice, not a data property — it is
`MIN_LENGTHSCALE_SPACINGS ×` the inducing pitch, which depends only on that
constant, `ninducing` and the extent. The figures here are at the 2.0 this file
was written under; at today's 0.25 both floors below divide by eight. At
production's 800,000 the floor is 1.78 km against a 250 m raster cell, and the
measured correlation is still 0.96 there, so only about 4% of the field's
variance falls below it. That part has nowhere to go: `construct_likelihood`
pins the noise at 1e-4 and sets `requires_grad_(False)`. At `ninducing` 12,000
the floor is 14.5 km and the unreachable fraction is 13%.

## Caveats

- One dataset, decimated to a tenth of production resolution, and `ninducing`
  12,000 against production's 800,000. See
  [the regime warning](#read-this-first-the-caps-are-calibrated-outside-the-production-regime).
- The uncapped outputscale grows with inducing density — 33 at 2,105 inducing
  points, 99 at 7,689 — so the runaway is worse at production scale than these
  numbers show.
- Four realisations per configuration. Sample minima and maxima over four draws
  are noisy: two identical runs at cap 16 gave sample maxima of 716 and 1330.
  $r$ is the stable statistic and still varies by about $\pm 0.1$ between
  identical runs.
- The uncapped outputscale had not converged at 250 epochs — it continues to
  climb to 750 (33 → 47 at 2,105 inducing points, 99 → 109 at 7,689). Quoted
  uncapped values are lower bounds.
- The measured correlation above is isotropic: `empirical_correlation` draws its
  step at a random angle and bins on distance alone. The raster shows
  north–south lineations, so a directional estimate would refine the 10.1 km
  half-correlation figure, and the kernel is isotropic too (`ard_num_dims=1`),
  so it could not follow a directional one.

## Reproducing

`docs/harness.py` trains one model per ladder and reports the trained kernel's
correlation function against the measured one. It trains via
`upscaling_tools.nngp.nngp_training` with the block's own
`construct_multiscale_kernel`, and `--sample-grid` samples via
`model.sample_gp(..., output_grid=GridSpec(...))` to exercise the FFT prior path
that production uses.

```
OMP_NUM_THREADS=1 python docs/harness.py \
    --train train.parquet --test test.parquet \
    --ninducing 100000 --epochs 250 --out results.json
```

`OMP_NUM_THREADS=1` is required: `faiss` and `torch` both load libomp, so the
first ELBO step segfaults on macOS without it.

The measurement primitives live in `upscaling_tools.spectral`, with tests in
`src/tests/test_spectral.py`:

| Function | Purpose |
| :--- | :--- |
| `empirical_correlation` | Correlation against lag, from binned products over point pairs. |
| `correlation_range` | The lag at which a curve falls to a given correlation. |
| `fit_ladder` | Additive Matern fit to a curve: lengthscales, variances, and the variance below the floor. |
| `bootstrap_ladder` | Spread of a fitted ladder under a spatial block bootstrap, for setting bound widths. |
| `kernel_correlation` | A fitted gpytorch kernel's own correlation against lag. |
| `spectral_misfit` | Mean absolute difference between two curves, integrated in log lag. |

`empirical_correlation` draws its pairs per bin rather than uniformly over the
point set. Uniformly drawn pairs land at a lag $h$ at a rate of
$(h / d_{\text{bbox}})^2$, so on this raster a 1 km bin takes about one pair in
a million and the short end of the curve is noise. That is the most likely
origin of the superseded autocorrelation table, and the property is guarded by
`test_empirical_correlation_covers_lags_far_below_the_domain`.

The earlier harness for the outputscale caps was a throwaway script and was
never committed, so the $r$ figures in this file cannot be reproduced directly.
`docs/harness.py` reports sample minima, maxima and standard deviations against
the observed ones instead, which needs no raster mask.

## Does the short-scale amplitude vary across the survey?

The kernel is stationary: one lengthscale and one outputscale for the whole
domain, and `core.py` fits one `predictive_nugget` for the whole domain. Both
assume the field is equally rough everywhere. `docs/tile_sweep.py` measures that
assumption.

It splits the domain into tiles and reports, per tile, the variance held in the
lag band $[\ell_{\text{short}}, s]$ — from the short component's floor up to one
inducing spacing, which is the band the model cannot resolve and the band the
nugget replaces with one number. The pair geometry copies
`empirical_correlation`: a random angle, a radius uniform in area, then the
nearest real point to where that lands. The band is **fixed across tiles**, so a
densely sampled tile cannot look smooth for a reason that has nothing to do with
the field. The value is absolute, not a fraction of the tile's own variance,
because a quiet tile and a rough one are the comparison being made and
`empirical_correlation` divides that difference out.

The spread across tiles, against the same run with the values shuffled across
locations:

| tiles | band variance, min | max | ratio | ratio, shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 2x2 | 0.0256 | 0.127 | 5.0 | 1.1 |
| 3x3 | 0.0102 | 0.175 | 17.2 | 1.2 |
| 4x4 | 0.0081 | 0.292 | 36.1 | 1.3 |

The amplitude varies, and the null test says it is the field rather than the
sampling. That test is load bearing here: the point spacing itself varies by a
factor of about 6,000 between tiles on this survey, so a nearest-neighbour
roughness would have measured the sampling instead. Under the shuffle every
ratio collapses to about 1.2 while the spacing map is unchanged.

The fitted short **lengthscale** is a separate question and this dataset does not
settle it. Its tile-to-tile ratio runs 6.9, 1.2 and 4.1 at the three tile counts,
against 1.9, 1.0 and 1.0 shuffled — above the null, but not stable enough to read
as a moving lengthscale. `fit_ladder` on a single tile also pins whole components
at zero variance, so the per-component variances are not reliable per tile; the
band variance above does not depend on that fit.

### Caveats

- VAL_GRAV_VARIOUS, 3,950 points, the same 80/20 split at seed 0 the rest of this
  file uses, at `ninducing` 2,000 giving 1,325 inducing points.
- Tiles below 200 points are skipped. At 4x4 that is most of them, so the 36.1
  compares a handful of survivors.
- One survey, far outside the production regime. See
  [the regime warning](#read-this-first-the-caps-are-calibrated-outside-the-production-regime).

### Reproducing the sweep

```
OMP_NUM_THREADS=1 python docs/tile_sweep.py \
    --train train.parquet --test test.parquet \
    --ninducing 2000 --tiles 2 3 4 --min-tile-points 200 \
    --out sweep.json
```

Add `--shuffle` for the null test, and `--model model_state.pth` to add the
held-out measurement: per tile, the coverage of the reported interval and the
`predictive_nugget` that tile would have fitted for itself. That is the
measurement that says whether the single global nugget costs anything, and it
needs a checkpoint trained to convergence to mean anything.

### What the block does with this

`kernel_options.modulation_control_count` turns the measurement above into a
model: it multiplies the short component by an amplitude field $g(x)$ built from
an $n \times n$ grid of RBF bumps, one trained weight each. The user README
gives the form. The long and mid components stay stationary, because the tile
sweep measures a short-band variance and says nothing about the regional trend.

The option is **off by default**, for two reasons.

- The sweep measures one survey, far outside the production regime, and on tiles
  that mostly hold too few points at 4x4. It says the short-band variance is not
  constant. It does not say a fitted $g$ recovers that variation, and no
  held-out comparison of a modulated fit against a stationary one has been run.
- The weights are new free parameters that raise prior variance wherever the
  ELBO can put it, which is the failure the outputscale caps exist to stop. The
  $[-1, 1]$ bound on each weight is the counterpart of those caps, chosen the
  same way and with no more evidence behind it.

A run that turns it on should be compared against the same run without it on
held-out error and interval coverage, and the two should be reported together.
The field and the `predictive_nugget` field both describe how rough a region is,
so a gain in one may simply be taken out of the other.

## The post-training health gate

The block used to retry only on a NaN. `_optim_step` raises `NaNTrainingError`
after `nan_patience` consecutive non-finite losses, and
`_train_with_noise_retries` caught it. A run that converged to a finite loss at a
bad optimum passed straight through, and the checkpoint was written as if the run
had succeeded.

The sweep at the `MIN_LENGTHSCALE_SPACINGS` table above threw exactly one such
run: at 2.0 spacings it gave `error_rate` 0.604, `predictive_nugget` 4.978 and
rmse 2.085, against 0.361 / 0.352 / 0.617 for the two healthy runs at the same
setting.

`_train_with_retries` now probes the fit after every attempt and rejects it on
either of two measurements.

### Measured on the block itself

`VAL_GRAV_VARIOUS.csv`, 3,160 training and 790 held-out points, `ninducing`
2,000 (1,317 realised), k 32, on CPU. At 60 epochs the first attempt lands on a
bad optimum and the second, from a new seed, does not:

| attempt | seed | training r2 | held-out r2 | nugget/var(y) | outcome |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | -0.518 | -0.514 | 1.272 | rejected |
| 2 | 1 | 0.926 | 0.904 | 0.063 | accepted |

Both gates fire on attempt 1 on their own. Before this change that attempt wrote
the checkpoint and logged "Model training completed".

At 250 epochs the first attempt is healthy and the gate passes it through:
training r2 0.850, held-out 0.809, nugget ratio 0.177 against the 0.5 threshold.

### The two gates

**`predictive_nugget` over `var(train_y)` at or above `MAX_NUGGET_RATIO`
(0.5).** This is the gate that does the work. At a ratio of 1.0 the correction
the reported interval needs is the whole variance of the data and the fit
explains nothing; 0.5 is where it explains less than it misses.

**Training `r2` at or below `MIN_TRAINING_R2` (-0.25).** A backstop, not the
main gate. It fires only when the held-out set holds fewer than
`MIN_CALIBRATION_POINTS` and the nugget ratio cannot be measured.

The held-out `r2` is not a gate at all. Good on training and bad on held-out is
hard data, not a broken optimiser. It is logged for context.

### Where the thresholds come from

`VAL_GRAV_VARIOUS.csv`, 3,160 training and 790 held-out points, k 32, 60 epochs,
CPU. Seventeen runs with the gates disabled, across four `ninducing` settings and
up to eight seeds, split into two populations with nothing in between:

| | training `r2` | nugget ratio | rmse |
| :--- | :--- | :--- | :--- |
| healthy, 12 runs | 0.712 to 0.943 | 0.062 to **0.389** | 0.73 to 1.45 |
| collapsed, 5 runs | -0.518 to **0.041** | **0.739** to 1.26 | 2.53 to 3.19 |

The nugget ratio separates every run. `MAX_NUGGET_RATIO` is set at 0.5, inside
the gap and clear of both ends.

**Why the training `r2` cannot be the main gate.** Its populations are separated
too, and by a much wider gap, but no value inside that gap transfers to another
survey. Anything above 0 is a claim about how much structure this particular
field has. A threshold at 0 is wrong in both directions at once:

- Too strict. A model that cannot resolve the field shrinks towards its mean
  function and scores near zero while behaving correctly. The reported interval
  is then widened by the predictive nugget, which is the honest answer, not a
  failure.
- Too lenient. The collapsed run at `ninducing` 100 scored **+0.041** and would
  pass, with an rmse 1.8x its healthy siblings.

At -0.25 it claims only what is safe: a fit that scores clearly below the mean of
the data is wrong rather than uninformative. It catches 2 of the 5 collapsed runs
on its own, which is all that is asked of a fallback.

### `error_rate_calibrated` is not usable here

An earlier reading of this was wrong and the sweep settled it. The collapsed runs
are calibrated: `error_rate_calibrated` came out at 0.043 to 0.051 across both
populations. `fit_predictive_nugget` widens the interval until the residuals fit
inside it, so a fit that explains nothing is honestly reported as explaining
nothing. The metric says the interval is trustworthy, not that the fit is good,
and it cannot separate the two populations.

### Why raw `error_rate` is not a gate

It is the number `fit_predictive_nugget` exists to correct, so it runs high on
healthy runs — 0.185 and 0.349 in the table above. A threshold anywhere near the
nominal 0.05 would fail them. It also separates worst: 1.7x between the
catastrophic run and its healthy sibling, against 3.4x for rmse and 14x for the
nugget.

### Why the retry reseeds rather than raising the noise

Noise growth is the lever for a NaN. It is the wrong lever here. The noise sits
inside the ELBO, so raising it changes the fit and makes it worse — the argument
in the `calibration.py` module docstring.

A run has exactly one source of randomness. `GridCountInducing.initialise` is a
regular grid masked by a cKDTree query, with no random draw, and the kernel
hyperparameters start at the `interval` midpoint. Only the mini-batch order is
random: `torch.randperm` in `nngp_training`, and gpytorch's own
`current_training_indices` on the inducing-indexed branch. Both follow the global
torch RNG, so `torch.manual_seed` per attempt is what changes the trajectory.

That also means the three runs per setting in the table above differed only by
mini-batch order, and that alone separated the two healthy runs from the
catastrophic one. It is direct evidence that a reseed escapes the bad optimum.

Because the inducing grid is deterministic, re-running the allocation per attempt
would return an identical grid. The allocation stays outside the retry loop.

### Cost

The probe runs `nngp_validation` with `plots=False` on `PROBE_POINTS` (5,000)
points per set. The figures are what cost, so the probe is a fraction of one
training epoch.

Each retry is a full training run. The worst case is `max_noise_retries` (5 by
default) plus `MAX_FIT_RETRIES` (2) plus the first attempt.

Seeding attempt 1 makes a run reproducible where it was not. A user who reruns a
bad config now gets the same bad result rather than a lucky reroll, which is the
point: the gate catches it either way.
