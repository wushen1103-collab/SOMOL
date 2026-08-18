# SOMOL

Sparse Overlap Measurement Operator Learning (SOMOL) calibrates a measured
bioactivity value from a named source assay to the scale of a named destination
assay. The implementation combines direct robust calibration, assay-graph
gauge transport, validation-only stacking, and support-bucket fusion.

This compact release reproduces the core ChEMBL 37 experiments for IC50, KI,
KD, and EC50 over grouped seeds 13, 17, 23, 29, and 31. It also includes the
exact no-gauge ablation, full-operator cycle audit, corrected resampled test,
hierarchical block bootstrap, and compact reference outputs used to verify the
reported results.

## Repository contents

- `scripts/`: data preparation, baselines, SOMOL selectors, ablations, and audits.
- `envs/environment.yml`: CPU-compatible reproducibility environment.
- `results/summary/`: compact aggregate results; no row-level bioactivity data.
- `tests/`: deterministic tests for the restored calibration and selector logic.
- `AUTHORS.md` and `CITATION.cff`: author, affiliation, and citation metadata.

Raw databases, processed records, prediction parquet files, model weights, and
logs are intentionally excluded because they are large and can be regenerated.

## Environment

Create and activate the environment with micromamba or conda:

```bash
micromamba create -f envs/environment.yml
micromamba activate somol
```

The main benchmark is CPU compatible. PyTorch is used only by the optional
neural affine baselines; the final SOMOL-RBB benchmark does not require a GPU.

## Data preparation

ChEMBL 37 is downloaded from the official EMBL-EBI release archive. The helper
prefers `aria2c`, then falls back to `wget` or `curl`.

```bash
python scripts/download_chembl.py \
  --release chembl_37 \
  --output-dir data/raw/chembl_37
```

The extracted SQLite file is normally located below
`data/raw/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db`.

## Full ChEMBL benchmark

The following command prepares all four endpoints, creates the five fixed
compound-target grouped splits, runs the comparator and SOMOL routes, refits
the exact no-gauge ablation, and produces cycle and dependence-aware inference
tables:

```bash
python scripts/run_multidataset_benchmark.py \
  --root . \
  --sqlite data/raw/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db \
  --endpoints IC50,KI,KD,EC50 \
  --seeds 13,17,23,29,31 \
  --workers 4
```

Use `--skip-existing` to resume a partially completed run. Use
`--skip-assay-context` to omit the assay-context surrogate comparators.
Generated data and predictions are written below `data/`, `results/benchmark/`,
and `results/ac_experiments/`; these paths are ignored by Git.

The final method-comparison module and exact no-gauge refit are implemented in
`run_method_comparison_baselines.py` and `run_exact_no_gauge_ablation.py`.
Their outputs have been checked against the archived experiment predictions to
floating-point precision.

## Result verification

The compact tables under `results/summary/` are sufficient to verify the main
endpoint means, the exact no-gauge penalties, cycle-return errors, and corrected
split-level inference without downloading row-level records:

```bash
python scripts/verify_release_results.py
pytest -q
```

## BindingDB audit

The large BindingDB Articles snapshot is not stored in Git. After placing the
public snapshot at the path accepted by the script, the source-publication
overlap audit can be rerun with:

```bash
python scripts/audit_bindingdb_publication_overlap.py --root .
```

Compact BindingDB dataset profiles, transport summaries, and DOI-overlap
results are included under `results/summary/`. They document the reported
cross-database reconstruction while avoiding redistribution of the source
snapshot and row-level measurements.

## Authors and citation

SOMOL is authored by Xiaohan Xu, Tianhao Zhang, Man Xi, Chenyu Zhao,
Martin Gluchman, Zhonghao Fan, Shuang Wang, Cong Zhang, and Hang Zhao.
Full affiliations, equal-contribution notes, and correspondence details are
provided in [AUTHORS.md](AUTHORS.md). Machine-readable citation metadata are
provided in [CITATION.cff](CITATION.cff).

## License

Released under the MIT License.
