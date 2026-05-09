# Evaluation consistency plan for raw and chunk loaders

## 1. Goal

This document plans how to make both raw-file evaluation and chunk-based
evaluation reliable and internally consistent across:

- validation during multi-GPU training;
- standalone multi-GPU testing;
- standalone single-GPU testing.

The target invariants are:

```text
raw eval -G1 global sample set
  == raw eval -GN global sample set
  == training-time raw val global sample set

chunk eval -G1 global sample set
  == chunk eval -GN global sample set
  == training-time chunk val global sample set

and

hist(raw -G1) == hist(raw -GN) == hist(training-time raw val)
hist(chunk -G1) == hist(chunk -GN) == hist(training-time chunk val)
```

If the global confusion matrix is identical, the final averaged metrics
(`mIoU`, `IoU`, and per-class IoU) are identical up to floating point
formatting.

This plan does not require raw-loader metrics and chunk-loader metrics to be
identical to each other. Raw and chunk loaders can still differ because their
input materialization paths differ. The required property is consistency
within each data loading mode.

## 2. Non-goals

- Do not change `tools/build_training_chunks.py` in this consistency pass.
- Do not regenerate existing chunks only to fix evaluation aggregation.
- Do not change the chunk payload schema unless a later validation proves it is
  unavoidable.
- Do not attempt to make chunk evaluation bitwise equivalent to the original
  raw-file loader in this plan. Cross-loader equivalence is a separate
  numerical-equivalence task.

## 3. Current risks

### 3.1 Metric state is not distributed-safe

`OccMetric.process()` currently accumulates a local confusion matrix in
`self.hist`, while `compute_metrics(results)` ignores `results` and reads
`self.hist` directly.

This is unsafe because:

- MMEngine normally collects `BaseMetric.results` across ranks, not arbitrary
  custom attributes.
- In multi-GPU evaluation, rank 0 may compute metrics from only rank 0's local
  subset.
- Repeated validation during training can be polluted if `self.hist` is not
  reset between evaluations.

### 3.2 Raw-loader multi-GPU evaluation is not guaranteed either

Raw-file configs use `NuScenesOccDataset` with `DefaultSampler(shuffle=False)`
for val/test. The sampler is responsible for rank partitioning, unlike chunk
configs where the dataset partitions internally.

However, raw-loader multi-GPU evaluation still uses the same `OccMetric`.
Therefore, even if `DefaultSampler` partitions samples correctly, the current
metric can still compute final metrics from only a local rank's `self.hist`.
Training-time raw multi-GPU validation has the same stale-state risk as chunk
validation.

Single-GPU standalone raw evaluation is the most trustworthy raw baseline
because all samples are processed in one process and no metric synchronization
is needed.

### 3.3 Chunk sample coverage is not audited

`NuScenesOccChunkDataset` partitions val/test samples by:

```python
global_offset % world_size == rank
```

This is a reasonable sample-level partition, but the current evaluation path
does not prove that all ranks together cover exactly the single-GPU sample set.
Missing samples, duplicated samples, empty ranks, or accidental second-stage
sampling would not necessarily fail fast.

### 3.4 Chunk `__len__` contract must be fixed

For val/test, `NuScenesOccChunkDataset.__len__()` currently returns the number
of samples handled by the current rank. This may or may not match the exact
assumptions made by MMEngine's evaluator and progress/collect logic for
`IterableDataset` with `sampler=None`.

This must be measured and then codified. It should not remain an implicit
assumption.

## 4. Design principles

1. **Samples are partitioned once.**
   Raw eval uses `DefaultSampler(shuffle=False)` for rank partitioning. Chunk
   eval uses `NuScenesOccChunkDataset` internal rank and worker partitioning.
   Do not combine both partitioning mechanisms in the same dataloader.

2. **Metrics are aggregated once in count space.**
   IoU must be computed after summing confusion matrices. Per-rank IoUs must
   never be averaged.

3. **Coverage errors are hard failures.**
   Duplicate or missing `sample_idx` records must raise an error when strict
   coverage checking is enabled. Silent de-duplication is not acceptable.

4. **Chunk generation remains unchanged.**
   The consistency fix belongs to evaluation protocol, metric aggregation, and
   validation diagnostics.

5. **Training-time val and standalone test use the same mode config.**
   Raw-mode comparisons use `configs/gausstr_featup.py` or
   `configs/gausstr_talk2dino.py`. Chunk-mode comparisons use
   `configs/*_chunks.py`. Within a comparison, the checkpoint, profile/split,
   evaluator, and relevant dataloader config must match.

## 5. Proposed implementation

### 5.1 Refactor `OccMetric` onto `BaseMetric.results`

Change `OccMetric` so `process()` appends one record per evaluated sample:

```python
self.results.append({
    "sample_idx": sample_idx,
    "hist": hist_i,
    "rank": rank,
})
```

`compute_metrics(results)` should:

1. read only the collected `results` argument;
2. validate `sample_idx` uniqueness;
3. optionally validate against an expected sample set;
4. sum all per-sample histograms into one global histogram;
5. compute `mIoU`, `IoU`, and per-class IoU from the global histogram;
6. report coverage diagnostics when enabled.

Important details:

- `sample_idx` must be a globally stable id from the dataset sample, not a
  local batch index.
- If the same `sample_idx` appears twice, fail in strict mode. Do not silently
  keep one copy.
- If expected sample ids are available and any are missing, fail in strict mode.
- The old persistent `self.hist` state should be removed or reset so repeated
  validation cannot accumulate stale counts.

This refactor is shared by raw and chunk modes. It is the main fix that makes:

- raw standalone multi-GPU test match raw standalone single-GPU test;
- raw training-time multi-GPU val match raw standalone test;
- chunk standalone multi-GPU test match chunk standalone single-GPU test;
- chunk training-time multi-GPU val match chunk standalone test.

### 5.2 Add strict coverage diagnostics

Add optional evaluator parameters such as:

```python
eval_debug=False
strict_coverage=False
expected_sample_idx_file=None
dump_coverage_prefix=None
```

The exact names can follow local style, but the behavior should cover:

- local rank sample count;
- local `sample_idx` hash;
- global sample count;
- global unique sample count;
- duplicate count and duplicate ids preview;
- missing count and missing ids preview, when expected ids are provided;
- global histogram hash.

In strict mode:

- duplicate samples fail;
- missing expected samples fail;
- unexpected samples fail;
- empty global result fails.

Diagnostics are not a substitute for correctness. They are there to make
failures easy to debug.

### 5.3 Codify chunk val/test `__len__`

For chunk val/test, `NuScenesOccChunkDataset.__len__()` should return the
global number of real samples, i.e. `manifest["num_samples"]` or
`index["num_samples"]`.

Rationale:

- MMEngine passes `len(dataset)` as the collection size to `BaseMetric`.
- Chunk val/test iteration is internally partitioned by rank, but the collected
  metric records represent the global validation set.
- Returning a rank-local length can cause collected metric results to be
  truncated to only a local subset.

Actual sample iteration remains rank-local; only the metric collection size is
global.

### 5.4 Preserve raw-mode sampler partitioning

Raw configs should keep the normal map-style dataset path:

```python
dataset=dict(type="NuScenesOccDataset", ...)
sampler=dict(type="DefaultSampler", shuffle=False)
drop_last=False
```

The metric refactor should rely on MMEngine's standard result collection and
dataset-size truncation behavior for map-style datasets. Coverage diagnostics
should still verify the final global `sample_idx` set, because sampler padding
or round-up behavior can otherwise hide duplicate samples.

### 5.5 Keep chunk dataset partitioning

Keep the current val/test rank partitioning policy:

```python
global_offset % world_size == rank
```

Keep worker partitioning inside the rank-local sample subset. This preserves:

- no duplicated work across ranks;
- no chunk-level imbalance requirement;
- full sample-level coverage even when `samples_per_chunk > 1`.

Configs should continue to use:

```python
sampler=None
drop_last=False
chunk_shuffle=False
sample_shuffle=False
skip_padding=True
```

for val/test.

## 6. Verification plan

### 6.1 Dataset coverage probe

Without running the model, collect sample ids yielded by the val/test
dataloader.

Required raw-mode checks:

- `-G1` sample set equals the raw val/test dataset sample set.
- Multi-GPU union equals the raw val/test dataset sample set after any
  framework-level size truncation.
- Duplicates introduced by sampler padding are either removed by standard
  collection/truncation before metric computation or fail strict coverage.
- Pairwise rank intersections are empty in the effective metric input.

Required chunk-mode checks:

- `-G1` sample set equals `index.json` sample set.
- Multi-GPU union equals `index.json` sample set.
- Multi-GPU pairwise intersections are empty.
- Per-rank counts sum to global count.
- Worker-level partitioning does not duplicate samples.

Run this for at least:

- `world_size=1`;
- `world_size=2`;
- `world_size=4` if available;
- a small subset where chunk count is not divisible by world size.

### 6.2 Metric aggregation unit test

Construct fake predictions, labels, masks, and sample ids. Verify that:

- one-process metric result equals multi-process metric result;
- duplicate sample ids fail in strict mode;
- missing expected sample ids fail in strict mode;
- two consecutive evaluations produce identical results and do not accumulate
  stale state.

### 6.3 End-to-end consistency test

Using the same checkpoint and raw configs, run:

```bash
PYTHONPATH=. mim test mmdet3d configs/gausstr_featup.py -C CKPT -G 1
PYTHONPATH=. mim test mmdet3d configs/gausstr_featup.py -C CKPT -l pytorch -G 2
PYTHONPATH=. mim test mmdet3d configs/gausstr_talk2dino.py -C CKPT -G 1
PYTHONPATH=. mim test mmdet3d configs/gausstr_talk2dino.py -C CKPT -l pytorch -G 2
```

Using the same checkpoint and chunk configs, run:

```bash
PYTHONPATH=. mim test mmdet3d configs/gausstr_featup_chunks.py -C CKPT -G 1
PYTHONPATH=. mim test mmdet3d configs/gausstr_featup_chunks.py -C CKPT -l pytorch -G 2
PYTHONPATH=. mim test mmdet3d configs/gausstr_talk2dino_chunks.py -C CKPT -G 1
PYTHONPATH=. mim test mmdet3d configs/gausstr_talk2dino_chunks.py -C CKPT -l pytorch -G 2
```

If more GPUs are available, also run `-G 4`.

For training-time validation, compare the validation result from the same epoch
checkpoint against standalone testing with the same mode config:

- raw training uses raw standalone test config;
- chunk training uses chunk standalone test config.

Acceptance criteria:

- global `sample_idx` set hash is identical;
- global histogram hash is identical;
- final metric values are identical within floating point formatting;
- running validation twice in the same process produces identical metrics.

## 7. Implementation order

1. Refactor `OccMetric` to use per-sample `BaseMetric.results`.
2. Set chunk val/test `__len__` to the global number of real samples.
3. Add strict coverage checking and diagnostic output for both modes.
4. Add a small probe or diagnostic to observe sample coverage for raw
   `DefaultSampler` evaluation and chunk `IterableDataset` evaluation.
5. Run raw and chunk dataset coverage probes.
6. Run metric aggregation tests.
7. Run raw-mode single-GPU and multi-GPU evaluation comparisons.
8. Run chunk-mode single-GPU and multi-GPU evaluation comparisons.
9. Only after both modes are internally trusted, optionally compare chunk eval
   against raw eval as a separate numerical-equivalence question.

## 8. Expert review

### 8.1 Review setup

Experts:

- `DistEval`: distributed evaluation engineer.
- `MMIntegrator`: MMEngine/MMLab integration expert.
- `RiskGuardian`: risk reviewer.
- `Pragmatist`: implementation pragmatist.

The review question was:

> Under the constraint of not modifying or minimally modifying chunk
> preprocessing, how should GaussTR guarantee that chunk-based training-time
> multi-GPU validation, standalone multi-GPU testing, and standalone single-GPU
> testing are trustworthy and produce identical final averaged metrics?

### 8.2 Round 1 independent reviews

`DistEval`:

- Do not modify chunk generation.
- Solve consistency in the evaluation partitioning and metric aggregation
  contracts.
- Samples must be partitioned once, and metrics must be aggregated once in raw
  count space.
- `OccMetric` must either all-reduce the confusion matrix or use standard
  `BaseMetric.results`; calculating per-rank IoU and averaging is wrong.

`MMIntegrator`:

- The current `OccMetric` bypasses `BaseMetric` by writing `self.hist` and
  ignoring collected `results`.
- The preferred fix is `process -> self.results -> compute_metrics(results)`.
- `IterableDataset.__len__` with `sampler=None` must be verified against
  MMEngine evaluator behavior.

`RiskGuardian`:

- Numeric closeness is not enough; evaluation must be auditable.
- Every entry point should expose sample ids, counts, de-duplicated counts, and
  per-rank counts.
- Missing samples, duplicate samples, stale metric state, rank-only logging,
  and hidden second-stage sampling are all unacceptable risks.

`Pragmatist`:

- Keep the chunk generation path unchanged.
- First prove sample coverage, then fix metric aggregation and reset behavior,
  then address `__len__`.
- Use a small fixed validation subset to run single-GPU, standalone multi-GPU,
  and training-time multi-GPU checks.

### 8.3 Author response

Accepted:

- Chunk generation remains out of scope for this consistency pass.
- `OccMetric` is the primary required fix.
- The preferred metric implementation should follow `BaseMetric.results`,
  not custom unsynchronized state.
- Coverage diagnostics should be part of the design, not an afterthought.
- Training-time validation and standalone testing must use the same chunk
  config when comparing results.

Refined:

- `all_reduce(self.hist)` is only a fallback. It can make counts aggregate, but
  it does not prove sample coverage. The preferred implementation stores
  per-sample `hist` records with stable `sample_idx`.
- `__len__` must be probed and then codified, not left implicit.

Rejected:

- Silently de-duplicating repeated `sample_idx` records. This could hide a
  partitioning bug. Strict mode should fail instead.

### 8.4 Round 2 cross-review

`DistEval`:

- Approved the `BaseMetric.results` direction.
- Required duplicate/missing coverage errors to be hard failures.
- Recommended per-sample histogram hashes across different world sizes.
- Warned that `__len__` behavior must be measured and fixed early.

`MMIntegrator`:

- Approved returning to the MMEngine metric contract.
- Required `sample_idx` to be globally stable.
- Required duplicate ids with conflicting histograms to fail rather than be
  ignored.
- Emphasized that `__len__` and id semantics must be fixed before relying on
  the path.

`RiskGuardian`:

- Approved the direction but did not fully release risk.
- Required code-level assertions for expected/seen/duplicate/missing coverage.
- Required tests where injected duplicate or missing samples fail.
- Rejected using de-duplication to silently repair partitioning mistakes.

`Pragmatist`:

- Approved the minimal-change direction.
- Recommended the concrete order: first probe and codify `__len__`, then
  refactor metric aggregation, then add coverage diagnostics, then run the
  acceptance matrix.
- Recommended saving sample-set diffs and histogram diffs on failure.

### 8.5 Final review conclusion

Consensus:

- Do not change chunk preprocessing for this task.
- The same `OccMetric` problem affects raw and chunk modes. Raw multi-GPU
  evaluation should not be assumed correct until the metric is refactored and
  coverage is verified.
- The evaluation consistency problem should be solved at the evaluator and
  dataset-evaluation contract layer.
- `OccMetric` must be changed to a distributed-safe, repeatable aggregation
  path.
- Sample coverage must be audited and must fail on missing or duplicate
  samples.
- Raw-mode sampler coverage and chunk-mode `__len__` behavior must be tested
  and codified.

Residual risk:

- Until MMEngine's collect behavior is verified for the chunk `IterableDataset`,
  `__len__` remains the main contract risk.
- Until MMEngine's raw-mode sampler padding/truncation behavior is verified,
  raw multi-GPU coverage remains a contract risk.
- Until strict coverage checks exist, identical final metrics alone are not
  enough proof of correctness.

Risk rating:

| Dimension | Before | After proposed plan | Notes |
| --- | --- | --- | --- |
| Multi-GPU metric correctness | High | Low-Medium | Low only after `BaseMetric.results` refactor and tests pass. |
| Raw sample coverage correctness | Medium | Low-Medium | Requires sampler coverage checks and duplicate/missing detection. |
| Chunk sample coverage correctness | High | Medium | Requires strict coverage checks and `__len__` contract validation. |
| Training-time val repeatability | High | Low-Medium | Requires metric state reset and repeated-eval test. |
| Chunk preprocessing impact | Low | Low | No chunk generation changes planned. |

### 8.6 Scope expansion note

The original expert review was triggered by chunk-mode inconsistency, but the
analysis generalizes because `OccMetric` is shared by raw and chunk configs.
The final implementation should therefore fix evaluation consistency for both
loading modes:

- raw mode: keep `NuScenesOccDataset` and `DefaultSampler`, but make metric
  aggregation distributed-safe and coverage-audited;
- chunk mode: keep `NuScenesOccChunkDataset` internal partitioning and
  `sampler=None`, then apply the same distributed-safe metric and coverage
  auditing.

With this design, it is feasible to make single-GPU standalone evaluation,
multi-GPU standalone evaluation, and training-time multi-GPU validation match
within each loading mode. It does not, by itself, guarantee raw-mode metrics and
chunk-mode metrics are identical to each other.
