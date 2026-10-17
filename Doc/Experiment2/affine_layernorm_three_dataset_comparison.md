# Affine LayerNorm Matching comparison

## Protocol

For CITE, EB, and Weinreb, Matching was initialized from scratch with seed 42 and trained for 30,000 steps on the GPU. Only the already-trained conditional path and Coupling were reused; no previous Matching weights or optimizer state were loaded. Width, depth, activation, loss, batch size, sigma, growth penalty, Coupling blocks, and learning-rate schedule follow the corresponding preceding non-affine run. The only architectural change is that each of the four field readouts uses `LayerNorm(elementwise_affine=True)`.

The preceding non-affine runs were warm-started, while these affine runs were trained from scratch as requested. Thus the comparison is an end-to-end comparison of these runs, but it does not isolate the affine parameters from initialization effects.

## SDE stability and mass fit

All 20 seeds (342–361), each using every initial data point, completed for every affine model. There were no failed rollouts or non-finite values. The maximum post-initial absolute coordinates were 141.83 (CITE), 58.27 (EB), and 96.57 (Weinreb); maximum absolute log weights were 10.70, 7.64, and 11.08 respectively.

Mean relative mass error by time, previous non-affine → affine-from-scratch:

| Dataset | Mean mass error by time |
|---|---|
| CITE | t=1: 0.24% → 0.56%; t=2: 1.88% → 1.00%; t=3: 8.27% → 14.02% |
| EB | t=1: 1.09% → 0.45%; t=2: 0.26% → 0.60%; t=3: 1.16% → 0.99%; t=4: 0.43% → 0.64% |
| Weinreb | t=1: 1.52% → 1.19%; t=2: 0.46% → 3.67% |

## W1 comparison

W1 was computed on all points for fixed SDE seeds 342 and 343. The mean terminal W1 across these two seeds was:

| Dataset | Previous non-affine | Affine from scratch | Change |
|---|---:|---:|---:|
| CITE | 32.080 | 35.468 | +10.6% |
| EB | 13.921 | 16.660 | +19.7% |
| Weinreb | 5.791 | 7.784 | +34.4% |

These results do not show a W1 improvement from the affine LayerNorm runs. Because their initialization protocol differs from the preceding warm-start runs, the degradation cannot be attributed to LayerNorm affine parameters alone.

## Interpretation

An affine LayerNorm immediately before a linear output head does not expand the representable function class. For hidden vector `h`, the affine form gives `W(γ ⊙ LN(h) + β) + b`, which is exactly another linear head with weight `W diag(γ)` and bias `b + Wβ`. It can change optimization dynamics, but does not inherently improve expressivity.

The learned affine parameters remained moderate: across the three models, the largest absolute scale was 2.49 and the largest absolute shift was 1.26. No affine-parameter explosion or SDE blow-up was observed. The main issue in these runs is reconstruction quality, especially the final CITE and Weinreb mass/W1 results.

For a strictly controlled causal comparison, the non-affine networks would also need fresh-from-scratch runs with the same seeds. The requested affine runs and all 20-seed rollout checks are complete.

## Artifacts

Each run has a `matching_initialization.json` confirming `warm_start: false`, the checkpoint architecture, and that no feedback checkpoint or optimizer state was loaded. Full per-seed rollout metrics are in each run's `evaluation_all_initial_points/checkpoint_architecture_20seeds_stability.json`.
