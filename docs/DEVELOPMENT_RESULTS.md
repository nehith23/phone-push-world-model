## Data improvements and earlier 1.5 cm evaluation

The latest processing independently estimates the table mapping in each frame with four reliably observed corner markers. Missing or ambiguous observations are rejected, never interpolated. Broad fixed spacing gates reject gross marker geometry errors. A 9/40 shaft-vector extrapolation estimates the contact point. Height, tilt and object-centre alignment remain approximate; framewise mapping can add detector jitter.

Cleaned original recordings supply 264 training, 61 validation and 58 reused-test transitions. The new eight-push pilot adds **160 training transitions**, for **424 training examples**. Original test clips and the inspected pilot are development data. Filtering disproportionately removes original bottom-to-top observations; only one validation transition remains in that direction. The retained-sample scores do not hide this coverage limitation.

All development controllers were rerun on the same revised 78.5 x 37 x 33 mm box:

| Controller | Reused targets reached | Mean final error |
|---|---:|---:|
| Geometric | 12/12 | 1.06 cm |
| Original MLP, unchanged weights | 6/12 | 2.59 cm |
| Earlier 11 mm MLP, unchanged weights | 8/12 | 2.65 cm |
| Cleaned original data only | 8/12 | 2.00 cm |
| Cleaned original data + pilot | 12/12 | 1.10 cm |

The old-only versus plus-pilot comparison isolates adding the pilot within the revised pipeline. It does not isolate every geometry/filtering change. [Full processing and development report](../results/stabilized/REPORT.md)

The combined model was then frozen before generating 24 new simulation scenarios:

| Controller | Fresh targets reached | Mean final error | Mean pushes |
|---|---:|---:|---:|
| Geometric | 24/24 | 1.02 cm | 3.75 |
| Learned world model | 22/24 | 1.12 cm | 4.79 |

The two learned failures remain in the average and are reported individually. These are fresh goals and initial states in the same simplified environment and parameter ranges, not fresh physical recordings, new objects, or camera-controlled manipulation. One seed and 24 cases do not establish a population-wide reliability rate. No simulator transitions trained the world model.

On the same 58 retained original development-test transitions, cleaned old-only and combined MLP position errors are 0.176 and 0.172 projected cm. Moving-only errors are 0.117 and 0.131; copying contact movement gives 0.118. Offline prediction is not uniformly improved.

