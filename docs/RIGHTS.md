# Attribution and publication rights review

## Source evidence

The thesis working directory is a clone of `https://github.com/BiDAlab/FairCVtest`, with inspected HEAD `0a12cdf`. Its tracked upstream tree contains `FairCV.py`, `README.md`, a dataset and two figures. There is no `LICENSE` file, SPDX identifier or explicit source-code redistribution grant in that snapshot. The README's “License” section is a dataset “AS IS” warranty/liability disclaimer, not a broad open-source grant. Public availability and citation requests do not resolve redistribution permission.

The early/mid/late PyTorch notebooks, prep/probe notebooks and helper file are local additions absent from upstream HEAD. That supports their status as the thesis implementation, but git additions alone are not legal proof of authorship. Architectures and experiment design build on Peña and Swati's prior work. The original helper says its feature construction is inspired by Swati's code; its two KL functions have ASTs identical to Peña's upstream functions. Comparison found no three-consecutive-line matches for the remaining helper functions against the locally available Peña/Swati source.

## Excluded rather than relicensed

- All `context/pena` and `context/swati` source, upstream `FairCV.py`, original upstream README and figures.
- All dataset/profile arrays, biography/name/face material, embeddings, tokenizers, weights and saved models.
- The two upstream-identical functions `kl_divergence` and `computeKL` from the original helper. The package imports a newly written vectorized compatibility estimator from `notebooks/legacy_kl.py` instead. Formula conventions, limitations and prior work remain attributed. This replacement is publication maintenance, not an original thesis contribution.
- Thesis PDF, environment/IDE metadata, original git history, notebook outputs, per-person examples and personal machine paths.

## Publication permission and reuse

The author confirmed permission to publish the local thesis additions and authorized public release of this package. Upstream source files and dataset assets remain excluded. This is a technical attribution review, not a legal opinion.

**No blanket open-source license is granted.** Publication for inspection does not relicense prior work or the underlying dataset. Contact the author for reuse permission for his contributions; retain the cited prior-work attribution.

FairCVdb and its constituent face/biography sources have separate terms. Anyone rerunning experiments must obtain and review those independently. No data download automation or data distribution is provided here.
