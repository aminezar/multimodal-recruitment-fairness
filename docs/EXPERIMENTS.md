# Experiment details and limitations

## Experimental conditions

| Condition | Tabular | Biography embeddings | Face embeddings |
|---|---|---|---|
| `none` | Competencies + gender | Original | Original |
| `tabular` | Competencies only | Original | Original |
| `tabular-text` | Competencies only | Blinded | Original |
| `tabular-image` | Competencies only | Original | SensitiveNets-processed |
| `tabular-image-text` | Competencies only | Blinded | SensitiveNets-processed |

The main mid-fusion text projection is 50-dimensional. The preserved sensitivity CSV covers widths 25, 35, 45 and 50; those extra runs are **not** part of the 75-run main matrix.

## Reproduce the static results preview — no dataset required

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python scripts/make_overview.py
python -m unittest discover -s tests -v
```

The chart script validates all 75 unique main run keys, then writes `assets/summary.csv` (means and **sample** standard deviations, denominator n−1, over five seeds) and `assets/results_overview.png`. Error bars are SD, **not** confidence intervals. These are calculations from archived metrics, not a new training run. PNG bytes were reproducible across two invocations in the tested environment; cross-version rendering can differ.

## Re-run the thesis experiments — separate, unverified workflow

1. Review the FairCVdb and constituent-data terms and obtain the dataset independently from the [upstream repository](https://github.com/BiDAlab/FairCVtest). Do not infer redistribution rights from its availability. Load only trusted NumPy files: the original loader uses `allow_pickle=True`, which can execute malicious serialized code.
2. Put `FairCVdb.npy` in a local, ignored `data/` directory. No dataset, biographies, faces, names, embeddings or model weights are distributed here.
3. Install `requirements.txt`. From `notebooks/prep/`, run `encode_bios.ipynb` to produce the four original/blinded train/test BERT arrays under `data/`. This uses frozen `bert-base-uncased` CLS embeddings, max length 256, and may download model weights. It was not executed in publication preparation.
4. From `notebooks/`, run `early_fusion.ipynb`, `mid_fusion.ipynb` and `late_fusion.ipynb`. Each saves its own run metrics to `results/`. The main export restricts mid-fusion to width 50. To repeat the sensitivity analysis, change its `TEXT_WIDTHS` and **use a distinct output filename**.
5. Build a separate combined CSV from the three newly generated files, with pandas `concat(..., ignore_index=True)`, and pass it as `python scripts/make_overview.py --input PATH_TO_NEW_CSV --output-dir PATH_TO_NEW_PREVIEW`. Do not overwrite archived `results.csv` unless intentionally replacing its provenance.
6. Input probes run from `notebooks/prep/`; the internal-representation probe runs from `notebooks/`. The label-baseline notebook reports only aggregate metrics in this export.

The historical setup uses 16 epochs, batch size 128, Adam defaults, L1 loss, hidden widths 40/20 and dropout 0.3. Helpers choose Apple MPS when available, otherwise CPU; CUDA selection is not implemented. Late fusion trains three modality models per experiment cell, so “75 runs” is an experimental-matrix count, **not** 75 neural-network fits. Seeds alone do not guarantee cross-device deterministic training.

## Metrics and limitations

- **KL:** historical male-to-female score-bin estimator, in natural-log units. It skips bins where either distribution has zero mass and excludes scores ≥0.98; it is not a general-purpose KL estimator. A new vectorized compatibility implementation replaces upstream-identical source functions; the archived numbers are unchanged.
- **Top-100 demographic parity:** smaller/larger gender selection-share ratio; 1 is ideal. `top100_female` is a proportion, not a percentage. The helper assumes at least 100 candidates and is not guarded against an empty selected gender group.
- **Equal opportunity:** male TPR minus female TPR; closer to 0 is better. Qualification uses the 75th percentile of blind training labels. The retained helper has a strict `>` prediction-threshold boundary, leaves exact ties unbinarized, and has no zero-denominator guard. These historical conventions are documented, not silently repaired.
- **NDCG@10:** ranking quality against blind labels; higher is better, not itself a fairness measure.
- **MAE:** lower is better relative to the specified biased or blind reference; units are score units, not percentage points.

FairCVdb is a **synthetic, controlled résumé benchmark**, but its biography and face sources include real-person-derived material. “Synthetic” does not eliminate privacy or licensing concerns. This package contains only code and experiment-level scalar outcomes. Findings do not establish real-world hiring fairness, causal effects, statistical significance or intersectional fairness. Gender is encoded as binary in this benchmark. Different debiasing methods, architectures and one dataset limit generalization.

## Citation

Amine Mohamed Zarouali. *Benchmarking Bias and Fairness in Multimodal AI Recruitment*. Bachelor’s thesis, Vrije Universiteit Brussel, August 2026. Supervisor: Prof. dr. Lynn Houthuys; advisor: Farnaz Faramarzi Lighvan. Thesis Sections 5–7 describe the experimental matrix, results and limitations. The thesis PDF is not redistributed in this package.

Prior work and data attribution:

- BiDAlab, [FairCVtest](https://github.com/BiDAlab/FairCVtest), inspected upstream snapshot `0a12cdf`.
- A. Peña et al. “Human-Centric Multimodal Machine Learning: Recent Advances and Testbed on AI-based Recruitment.” *Springer Nature Computer Science*, 4, 434, 2023.
- A. Morales, J. Fierrez, R. Vera-Rodriguez and R. Tolosana. “SensitiveNets: Learning Agnostic Representations with Application to Face Images.” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020.
- M. De-Arteaga et al. “Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting.” *FAccT*, pp. 120–128, 2019.

No blanket open-source license is asserted. The upstream snapshot has no explicit source-code redistribution grant; third-party source and dataset assets are excluded. Publication permission and reuse terms are described in [RIGHTS.md](RIGHTS.md).
