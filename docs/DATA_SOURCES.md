# Data and model sources

## Study-produced artifacts

The released per-query outcomes, rankings, numerical arrays, summaries and figure data are derived research outputs. Their source locations and SHA-256 hashes are in `provenance/artifact_sources.json`. Raw benchmark document text, query text and model weights are acquired from their respective sources.

## ExcluIR

The official project is https://github.com/zwh-sdu/ExcluIR. Its README links the original dataset download. The local study used 3,452 requests and a 90,406-document corpus. On 2026-11-09 the upstream repository root contained README.md and src, and the GitHub license field was null. Dataset redistribution terms remain to be clarified through the original provider.

## IFIR

The official dataset is https://huggingface.co/datasets/songtingyu/IFIR. The study pins revision `0e1bd15954eed1414023961a9f79038cf2aad2c6`. The pinned dataset card declares MIT. The GitHub project https://github.com/SighingSnow/IFIR provides project context. Dataset-specific underlying corpus attribution and terms also apply.

Pinned source downloads follow this pattern.

```
https://huggingface.co/datasets/songtingyu/IFIR/resolve/0e1bd15954eed1414023961a9f79038cf2aad2c6/fiqa/fiqa-query.jsonl
```

FiQA and NFCorpus form the original equal-collection transfer mean. The later CDS evaluation includes all 43 selected requests and is reported separately.

## Frozen models

MiniLM uses https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Qwen uses https://huggingface.co/Qwen/Qwen3-Reranker-4B at revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`. Their original cards and model terms apply.
