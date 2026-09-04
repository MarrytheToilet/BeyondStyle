# BeyondStyle server runbook

This is the canonical migration sequence. It intentionally leaves the
PyTorch command and model Hub IDs unresolved until the server's GPU, driver,
CUDA compatibility, storage, and network access are known.

## 1. Inspect the server before installing anything

Record:

```bash
nvidia-smi
python3 --version
df -h
```

Use a new Python 3.10 or 3.11 virtual environment dedicated to BeyondStyle.
Do not reuse an environment containing unsloth, torchao, TransformerLens, or
sentence-transformers: those packages impose unrelated and often conflicting
Torch/Transformers constraints.

## 2. Install an isolated runtime

Install the official PyTorch wheel matching the server driver and CUDA
compatibility first. Then install only the project layers needed by the job:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the server-compatible PyTorch wheel here.
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/evaluation.txt
python -m pip install -r requirements/analysis.txt
python -m pip install -r requirements/dev.txt
python -m pip check
```

FlashAttention is optional. Install it only after Torch is fixed, and only
when a wheel/source build matches that exact Torch and CUDA ABI. The project's
`auto` policy uses the profiled, portable PyTorch SDPA path; select
`--attention-backend flash_attention_2` only after benchmarking that exact
model and GPU. External FlashAttention 2 is slower for the locally tested
Qwen3/Qwen3.5 models and cannot run Gemma 4 heads wider than 256 dimensions.

## 3. Synchronize code and experiment inputs

The repository, proposition-aligned pairs, frozen prompt files, and their
manifests are required. Raw corpora and existing outputs are optional.
Never copy `.env` through a shared command or commit it.

Example from the local machine:

```bash
rsync -az --delete \
  --exclude '.env' \
  --exclude '__pycache__' \
  --exclude 'outputs/' \
  --exclude 'data/raw/' \
  /path/to/BeyondStyle/ user@server:/path/to/BeyondStyle/
```

`--delete` is appropriate only for a newly created dedicated server copy.
For an existing remote workspace, omit it until the remote contents have been
audited.

## 4. Download models outside the repository

Use one server model root, for example `/data/models`, and keep weights
outside the code directory. After authentication through the Hugging Face
CLI, download each agreed model snapshot into an explicit local directory.
Do not place access tokens in shell history.

The retained target set is:

- English primary: Llama-3.1-8B-Instruct and Qwen3-4B.
- English robustness: gemma-4-E2B-it.
- Chinese replication: Qwen3-4B, Qwen3.5-4B and gemma-4-E2B-it.

Run configuration-only validation before loading weights:

```bash
python scripts/check_environment.py \
  --require-cuda \
  --pip-check \
  --model /data/models/Qwen3-4B \
  --model /data/models/gemma-4-E2B-it \
  --json-out logs/server_environment.json
```

## 5. Run the real-model smoke test

Use the smallest downloaded causal LM:

```bash
python scripts/smoke_steering_runtime.py \
  --model /data/models/Qwen3-4B \
  --max-new-tokens 4 \
  --json-out logs/steering_runtime_smoke.json
```

The report must show:

- `position_mode = decode_only`;
- `prompt_prefill_untouched = true`;
- exactly one skipped prefill call for each nonzero condition;
- no hook for the zero-strength baseline;
- the resolved dtype and attention backend.

Before reserving a GPU for downstream evaluation, prefetch task data on CPU
and then verify that the complete task set works offline:

```bash
python scripts/check_lm_eval_data.py \
  --json-out logs/lm_eval_data_prefetch.json

python scripts/check_lm_eval_data.py \
  --offline \
  --json-out logs/lm_eval_data_offline_check.json
```

The default check covers the ten-task fast grid: PIQA, WinoGrande, Social IQa,
COPA, WiC, RTE, BoolQ, OpenBookQA, SciQ, and ARC-Challenge. Use
`eval_lm_benchmark.py --task-group fast --limit 100` across all
models/strengths, raise the cap to 200 for shortlisted conditions, and remove
the cap only for the final selected conditions. The same tasks are exposed as
`science_physical_world` (PIQA, OpenBookQA, SciQ, ARC-Challenge),
`human_social_world` (Social IQa, COPA, WinoGrande), and
`language_reading_control` (WiC, RTE, BoolQ). Treat the third family as a
domain-light language control when interpreting cross-family differences.

## 6. Launch retained source-direction jobs

All long jobs should run in `tmux`, with unbuffered output and one process per
GPU. A single process owns one model and reuses it across layers,
representations, prompts, and steering strengths.

The formal runner defaults to `--cache-implementation dynamic`,
`--pad-to-multiple-of 8`, and descending prompt-length batching. On the
validated PyTorch 2.8 / Transformers 5.14 stack, StaticCache can trigger
repeated `torch.compile` specialization and empty CUDA-graph captures when
decoder hooks are installed. StaticCache is opt-in and should only be used
after a model-specific smoke test demonstrates a real throughput gain and
leaves no compilation workers behind.

```bash
export MODEL_ROOT=/data/models
export EXTRACTION_BATCH_SIZE=4
export BATCH_SIZE=4

tmux new -s beyondstyle-en
python -u scripts/run_hidden_formal.sh en 2>&1 | tee logs/hidden_en.log
python -u scripts/run_steering_full.sh en 2>&1 | tee logs/steering_en.log
```

Run Chinese in a separate session or on a separate GPU. Increase batch sizes
only after the smoke test and a short real batch. Extraction halves its batch
on CUDA OOM; formal generation atomically splits failed shards and resumes
without regenerating completed shards.

After document states and nullspace checkpoints are present, the token-level
localization jobs can use one model per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/analyze_token_projections.py \
  --model "$MODEL_ROOT/Llama-3.1-8B-Instruct" \
  2>&1 | tee logs/token_projection_llama.log

CUDA_VISIBLE_DEVICES=1 python -u scripts/analyze_token_projections.py \
  --model "$MODEL_ROOT/Qwen3-4B" \
  2>&1 | tee logs/token_projection_qwen.log
```

The default is leave-one-generator-out nullspace fitting. Only scalar token
coordinates are saved as resumable Parquet shards; full token hidden states
are never written.

`run_steering_full.sh` now defaults to the symmetric source-direction pair
\(\beta=-1,+1\). Wider strength grids are development diagnostics and must not
be mixed into confirmatory outputs.

Extended audits additionally evaluate source/explained/residual/random component
steering. Do not launch that stage until the automatic linguistic
feature extractor and decomposition checkpoints have passed their local smoke
tests.

## 7. Preserve evidence

For every completed run, retain:

- environment check and smoke-test JSON;
- stdout/stderr logs;
- state and checkpoint hashes;
- generator-held-out layer-selection table;
- steering calibration source and scale;
- configuration fingerprint;
- final result hash and throughput statistics.

The formal generation runner writes these fields into its manifest. A work
directory with a different configuration fingerprint is never silently
reused.

If Pangram is run on the server, keep its key in the private server `.env` and
retain the SDK version, scoring timestamp, input hash, request failures and raw
responses. Detector output is auxiliary and must never control job selection.

## 8. Cross-model audit bundle

The retained GPU audit is bounded to three questions:

1. **Direction-family comparison:** run the encoding pattern and the
   nullspace-projected source difference under the same 166-prompt,
   three-paired-seed protocol as the canonical pullback.
2. **Decoding robustness:** evaluate the frozen pullbacks under greedy
   decoding and temperature 1.0, alongside the existing temperature-0.7
   three-seed run.
3. **Third-family replication:** repeat prefix-only readout transfer,
   the 100-pipeline fixed-token null, and the three-seed behavior audit in
   Gemma 4 E2B using the same relative intervention checkpoint, final
   endpoint, nuisance rank, fit count, 5% state norm, and sealed prompts.

No broad layer-by-strength grid or 70B model is part of this audit bundle.
The two-GPU scheduler runs Llama followed by Gemma on GPU 0
and Qwen on GPU 1:

```bash
export MODEL_ROOT=/root/autodl-tmp/models
tmux new-session -d -s beyondstyle-audit \
  "cd /root/autodl-tmp/BeyondStyle && \
   MODEL_ROOT=\$MODEL_ROOT bash scripts/run_server_revision_bundle.sh"
```

The bundle records the complete runtime identity before and after the run,
performs Gemma layer/hook smoke tests before the expensive stages, writes
resumable generation shards, and rebuilds optional summary tables when
their source CSVs exist. Based on the recorded Llama/Qwen throughput, budget
roughly 2.5--4 wall-clock hours and less than 2 GB of new artifacts; Gemma
compatibility failures stop before its full causal audit rather than
silently changing the protocol.
