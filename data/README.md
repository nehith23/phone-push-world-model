# Data and provenance

I recorded all five videos in `raw/` myself on a phone. They are stored byte-for-byte as recorded. [raw_manifest.json](raw_manifest.json) records their SHA-256 hashes and sizes; the extraction report retains the source hashes too.

| Video | Direction of push | Reviewed pushes | Role |
|---|---|---:|---|
| left push.mp4 | Left to right | 6 | Original whole-push split |
| right push.mp4 | Right to left | 6 | Original whole-push split |
| top push.mp4 | Top to bottom, toward camera | 6 | Original whole-push split |
| down push.mp4 | Bottom to top, away from camera | 7 | Original whole-push split |
| calibration_pilot_v2.mp4 | Two per direction | 8 | Training and development only |

Video names describe the approach side. Manual resets are excluded using annotated push windows. Original pushes 1–4 train, push 5 validates and push 6 is the original test; the seventh down push also trains. The pilot adds training examples. Splitting keeps complete pushes together, but the original test clips have since been inspected and used in development. They are not a fresh physical-session test.

## Current measurements

The workspace is approximately 400 x 300 mm between marker centres. The cover is 78.5 x 37 x 33 mm, the roof-marker spacing approximately 30 mm, the tool-marker spacing approximately 40 mm, and front-tool-marker-to-contact offset 9 mm. [Remeasured setup](remeasured_setup.json) is the current source. [manifest.json](manifest.json) preserves older estimates for historical experiments; do not apply its original dimensions to the final results.

The cover is a hollow rigid body without wheels. The pusher-marker height was not measured, the caliper was not consistently flat, and the alignment of the roof-marker midpoint with the physical object centre is unconfirmed. Table-plane projections therefore remain approximate.

## Retained data

Framewise mapping requires four unambiguous corner observations. Broad spacing gates and displacement checks reject invalid observations; rejected frames are not interpolated. The final extraction retains 543 transitions: 264 original training + 160 pilot training, 61 validation and 58 reused-test. Adjacent-frame differences are correlated; 424 transitions are not 424 independent demonstrations.

Filtering is uneven: only 20 original bottom-to-top transitions survive across all splits, with one in validation. The pilot contributes 37 additional bottom-to-top training transitions. Framewise mapping compensates for camera movement under planar assumptions but can also add detector jitter. No independent metric ground truth validates physical millimetre accuracy.

## Rebuild

In a separate project copy, run:

```powershell
python scripts/stabilize_dataset.py --video-dir data/raw
python scripts/train_stabilized.py
```

Paths above are relative to the repository root. Rebuilding writes datasets and checkpoints. Follow the [reproducibility guide](../docs/REPRODUCIBILITY.md) to preserve frozen submission evidence.
