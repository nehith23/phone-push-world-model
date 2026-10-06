# Results index

Only two folders matter for the final claims. Everything else is preserved development history, kept because the frozen protocols and earlier reports point to it.

| Folder | What it is | Status |
|---|---|---|
| `precision/fresh/` | Final paired evaluation: 24 new scenarios, geometric vs learned, 5 mm tolerance | **Final result** |
| `precision/demo/` | Final controller video and its replay verification | **Final result** |
| `stabilized/` | Cleaned dataset, trained checkpoints (`plus_pilot/small_mlp.pt` is the final model) | Final data and model |
| `reproduction/` | Fresh-environment rerun logs and checks | Verification |
| `demo/` | Recorded-motion replay: phone trajectory retargeted to the Panda (no learning) | Demo |
| `tracking/`, `simulation/` | First tracking pass and first replay | History |
| `baseline/`, `goal_control/`, `contact_baseline/`, `contact_proxy/`, `diagnosis/`, `pilot_v2/`, `frozen_evaluation/`, `precision/` (top level) | Earlier experiments, each with its own `REPORT.md` | History |

The order of earlier experiments and what each changed is in [docs/EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md) and [docs/DEVELOPMENT_RESULTS.md](../docs/DEVELOPMENT_RESULTS.md).
