# Verification evidence

## Executed in a scratch environment

Python 3.11.15; core dependency versions in `requirements-tested.txt`. The original `.venv` repeatedly timed out during scientific-library imports; the machine's Anaconda environment had NumPy/pandas binary incompatibility. Neither environment was modified. A separate scratch venv was used successfully instead; it is not part of the package.

From the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/make_overview.py
```

Publication suite: **5 tests passed**. Scope:

- Seven notebooks validate as nbformat 4; all code cells compile; no outputs/execution counts/global or cell metadata remain.
- Early, mid and late model classes extracted from the notebooks each execute a bounded forward pass with finite [0,1] outputs.
- Helper metric checks for MAE, top-100 parity, EO and NDCG.
- Five feature conditions and the gender loader exercised against explicit temporary synthetic fixtures, not benchmark data.
- Training helper exercised for one CPU epoch on eight synthetic examples, batch size four; four finite predictions returned. This is a smoke test, not a thesis result.
- Chart validates the unique 75-run matrix, writes 120 summary rows, computes sample SD, and produces byte-identical PNGs across two invocations in the tested environment.
- Independent KL estimator exercised against a hand-computed legacy boundary example.

Separate local audit: 600 scalar comparisons between canonical results and original separate early/late/mid-width-50 CSVs agree within 1e-12. The replacement KL implementation matches the original local formula on 101 random/boundary comparisons within 1e-13. The reference implementation was used only locally, not exported. Supplemental CSVs have five unique seeds per group.

The PNG was visually inspected: six labeled panels, distinct fusion colors/markers, explicit metric units/direction/ideal values, readable legends and SD/benchmark caution captions. It is a static image, not an interactive site.

## Not verified

No full thesis retraining, data-array loading, BERT extraction or download, full probe-classifier fits, MPS training, CUDA training, cross-device determinism or original-environment reproduction was attempted. Historical EO threshold edge cases and KL conventions are retained/documented rather than changed. The source project's full training pipeline has no claimed passing test suite; the above is the **new publication-focused suite**.

## Publication gate

The author confirmed permission to publish his local thesis additions; see `RIGHTS.md`. Publication includes only the intentional package files, excluding the scratch venv, audit logs, source inventory, original repository, thesis PDF and local machine paths. The local source audit checked unchanged original-tree hashes/git status and excluded forbidden assets. The publication-focused tests and chart generation were rerun successfully before upload.
