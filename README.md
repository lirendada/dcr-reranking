# DCR reranking artifacts

Code and derived research data accompanying **Standalone Reranking Gains and Fusion Value for Instruction-Bearing Queries**.

Manuscript authors are Haoheng Chen, Xiaodong Song, Congyang Cheng and Haohan Ding. Haohan Ding is the corresponding author. The manuscript is under author review.

## Quick start

Python 3.10 or newer is sufficient to check artifact integrity and reconstruct point metrics from the published per-query outputs.

```bash
python scripts/check_artifacts.py
python scripts/replay_results.py --output outputs/point_metrics.json
```

The replay checks every published per-query metric group against the archived experiment summaries. Output values use percent units. It also reconstructs the original equal-collection FiQA/NFCorpus point mean while keeping CDS separate.

For the saved bootstrap-array checks and figures, install the recorded analysis dependencies in your own environment.

```bash
python -m pip install -r requirements.txt
python scripts/check_bootstrap.py
python figures/draw_figures.py
```

## Contents

- `data/per_query/` contains study-produced evaluation outcomes for the source inputs, fusion variants and transfer analyses.
- `data/arrays/` contains archived source attribution and input-substitution draws and head orders.
- `data/cds/` contains the archived CDS feature array, draws and ranking outputs.
- `data/q7_rankings/` contains the archived Q7 target ranking outputs.
- `data/summaries/` contains archived experiment summaries and interval roles.
- `kernels/` contains the original numerical feature and pairwise-ridge kernels.
- `experiments/` contains scientific experiment source snapshots, with generic machine paths. These drivers require the original intermediate-artifact layout.
- `figures/` contains the plotting code and numerical figure data.
- `provenance/` records source hashes and the complete Q7 cache dependency list.

See [data sources](docs/DATA_SOURCES.md), [reconstruction scope](docs/REPRODUCIBILITY.md), and [manuscript mapping](docs/PAPER_TABLE_MAP.md).

## Interpretation and reconstruction scope

ExcluIR was repeatedly used for development. FiQA/NFCorpus follow-ups reused observed targets. Q7 is a post hoc feature control. All variants retain MiniLM-topic head selection, and CDS remains separate from the original two-collection mean.

The public quickstart reconstructs saved-outcome point estimates. The saved-draw checks reconstruct selected percentile intervals. Repeating the original source refits and upstream neural scoring additionally requires the intermediate feature and score caches listed in `provenance/full_cache_requirements.json`. Export of that complete cache is pending. Original retrieval and neural inference have a separate execution scope.

## Licenses

Original code is released under [MIT](LICENSE). Study-produced data use [CC BY 4.0](data/LICENSE.md). Upstream benchmarks and pretrained models retain their source licenses and attribution requirements.
