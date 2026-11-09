# Reconstruction scope

## Available entry points

`check_artifacts.py` checks the released file hashes. `replay_results.py` reads the released per-query records, checks unique query/method units and metric ranges, recomputes group means and compares them with archived summaries. `check_bootstrap.py` recomputes selected source point differences and percentile intervals from archived draw arrays.

The frozen numerical kernels are available in `kernels/`. Scientific preparation, inference, fitting and scoring drivers are archived under `experiments/`. Machine-specific home paths are generalized in these source snapshots. The original source hash and the published source hash are retained in provenance.

## Additional original-cache requirements

The full Q7 reconstruction depends on the 76 recorded inputs in `provenance/full_cache_requirements.json`, including source features, fold views, target feature arrays and saved neural scores. The public package currently includes the available study-produced outcomes and selected numeric arrays. Complete intermediate-cache export remains pending access to the original study archive.

The historical full cache replay used Python 3.12.3 and NumPy 2.2.6. It performed no new neural inference. Current public quickstart checks are recorded separately in the local publication record. Empty-environment upstream retrieval, neural-score generation and full refitting have a separate validation scope.

## Scientific limits

Standalone input gains and incremental feature value are distinct outcomes. Repeated development exposure, post hoc Q7 specification, observed target follow-ups, collection-specific request construction, limited CDS coverage, and jointly differing model configurations remain part of the interpretation. Numerical consistency supports the stated computational checks.
