# DiffusionGemma Benchmarking with `lm-evaluation-harness`

This setup is intended for benchmarking **`google/diffusiongemma-26B-A4B-it`** with an editable Hugging Face Transformers installation and a custom DiffusionGemma backend for `lm-evaluation-harness`.

The editable Transformers installation allows direct modification of DiffusionGemma internals such as attention, masking, architecture, self-conditioning, and sampling.

---

## Project Structure

```text
~/diffusiongemma-bench/
├── transformers/
├── lm-evaluation-harness/
├── results/
└── test_diffusion_gemma.py
```

---

# 1. Create the Conda Environment

```bash
conda create -n diffusiongemma-eval python=3.11 -y
conda activate diffusiongemma-eval

python -m pip install --upgrade pip setuptools wheel
pip install torch
```

Create the project directory:

```bash
mkdir -p ~/diffusiongemma-bench
cd ~/diffusiongemma-bench
```

---

# 2. Install Editable Transformers

Clone the custom Transformers repository:

```bash
git clone git@github.com:mojtaba-nafez/transformers-gemma-dlm.git
mv transformers-gemma-dlm transformers

cd transformers
pip install -e .
pip install accelerate
```

Because Transformers is installed in editable mode, changes made inside:

```text
~/diffusiongemma-bench/transformers/src/transformers/
```

will be used directly without reinstalling the package.

Return to the project root:

```bash
cd ~/diffusiongemma-bench
```

---

# 3. Install Editable `lm-evaluation-harness`

Clone the custom evaluation repository:

```bash
git clone git@github.com:mojtaba-nafez/lm-evaluation-harness-gemma-dlm.git
mv lm-evaluation-harness-gemma-dlm lm-evaluation-harness

cd lm-evaluation-harness
pip install -e ".[hf]"
```

Return to the project root:

```bash
cd ~/diffusiongemma-bench
```

---

# 4. Verify Editable Installations

Run:

```bash
python - <<'PY'
import transformers
import lm_eval

print("Transformers:")
print(transformers.__file__)

print("\nlm_eval:")
print(lm_eval.__file__)
PY
```

Expected output should point to the local editable repositories:

```text
Transformers:
/home/.../diffusiongemma-bench/transformers/src/transformers/__init__.py

lm_eval:
/home/.../diffusiongemma-bench/lm-evaluation-harness/lm_eval/__init__.py
```

---

# 5. DiffusionGemma `lm-eval` Integration

The custom `lm-evaluation-harness` repository already contains:

```text
lm_eval/models/diffusion_gemma_hf.py
```

This file provides the Hugging Face DiffusionGemma adapter used by `lm-evaluation-harness`.

The model is registered in:

```text
lm_eval/models/__init__.py
```

under the name:

```text
hf-diffusion-gemma
```

Confirm that the model registration works:

```bash
python - <<'PY'
from lm_eval.api.registry import get_model

model_cls = get_model("hf-diffusion-gemma")

print(model_cls)
PY
```

---

# 6. Check the Model

Before running a benchmark, verify that DiffusionGemma loads and generates correctly:

```bash
cd ~/diffusiongemma-bench
python test_diffusion_gemma.py
```

Make sure this works successfully before starting MMLU-Pro evaluation.

---

# 7. MMLU-Pro Smoke Test

Start with only five examples from the Biology subset:

```bash
cd ~/diffusiongemma-bench

mkdir -p results

lm_eval \
    --model hf-diffusion-gemma \
    --model_args pretrained=google/diffusiongemma-26B-A4B-it,dtype=bfloat16,parallelize=True,max_length=8192 \
    --tasks mmlu_pro_biology \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --limit 5 \
    --log_samples \
    --output_path results/mmlu_pro_biology_smoke
```

This small run is useful for verifying:

- model loading
- DiffusionGemma generation
- MMLU-Pro prompt formatting
- answer extraction
- result logging
- GPU memory usage

Inspect the generated samples before running the full benchmark.

---

# 8. MMLU-Pro Smoke Test with Multiple GPUs

If multiple GPUs are available, enable model parallelization with:

```text
parallelize=True
```

Run:

```bash
cd ~/diffusiongemma-bench

lm_eval \
    --model hf-diffusion-gemma \
    --model_args pretrained=google/diffusiongemma-26B-A4B-it,dtype=bfloat16,parallelize=True,max_length=8192,rm_self_attention=True \
    --tasks mmlu_pro_biology \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --limit 5 \
    --log_samples \
    --output_path results/mmlu_pro_biology_smoke
```

For the first experiments, keep:

```text
batch_size=1
```

until the full pipeline is confirmed to work correctly.

---

# 9. Run Full MMLU-Pro

After confirming that the smoke test works correctly:

```bash
cd ~/diffusiongemma-bench

lm_eval \
    --model hf-diffusion-gemma \
    --model_args pretrained=google/diffusiongemma-26B-A4B-it,dtype=bfloat16,parallelize=True,max_length=8192 \
    --tasks mmlu_pro \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path results/diffusiongemma_mmlu_pro_baseline
```

For a multi-GPU system, add:

```text
parallelize=True
```

to `--model_args`.

Example:

```bash
lm_eval \
    --model hf-diffusion-gemma \
    --model_args pretrained=google/diffusiongemma-26B-A4B-it,dtype=bfloat16,parallelize=True,max_length=8192 \
    --tasks mmlu_pro \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path results/diffusiongemma_mmlu_pro_baseline
```

```bash
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gemma_diffusion/main.sh
```
```bash
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gemma_diffusion/main2.sh
```


---

# 10. Editable DiffusionGemma Development

The Transformers repository is installed in editable mode, so modifications to the DiffusionGemma implementation are immediately visible to the benchmark process.

The relevant source directory is:

```text
~/diffusiongemma-bench/transformers/src/transformers/models/diffusion_gemma/
```

This allows experimentation with areas such as:

- attention behavior
- attention masks
- bidirectional decoding
- self-conditioning
- sampling strategy
- denoising schedule
- entropy-based acceptance
- token confidence
- re-noising
- convergence criteria
- architectural ablations

After modifying the source, rerun the same benchmark command to compare against the baseline.

---

# Recommended Workflow

```text
1. Install the editable repositories
        ↓
2. Verify local package paths
        ↓
3. Run test_diffusion_gemma.py
        ↓
4. Run 5-example MMLU-Pro Biology smoke test
        ↓
5. Inspect generated samples
        ↓
6. Run complete MMLU-Pro baseline
        ↓
7. Modify DiffusionGemma
        ↓
8. Rerun exactly the same benchmark
        ↓
9. Compare against the baseline
```

Keep the benchmark settings fixed when evaluating architecture or sampling changes so that differences can be attributed to the model modification rather than changes in the evaluation configuration.