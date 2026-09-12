# ── Shared environment ────────────────────────────────────────────────────────
# One .venv at ml-scratchpad/ root. Run `make venv` once, then `make kernel`.

.PHONY: venv install kernel deps clean nanogpt-notebook nanogpt-regen kv-cache-notebook kv-cache-regen rope-notebook rope-regen inference-engine-notebook inference-engine-regen continuous-batching-notebook continuous-batching-regen turbo-quant-notebook attention-transformer-notebook turbo-quant-regen attention-transformer-regen additive-inference-notebook additive-inference-regen vllm-memory-notebook vllm-memory-regen vllm-memory-html turbo-quant-html attention-transformer-html nanogpt-html kv-cache-html rope-html inference-engine-html continuous-batching-html additive-inference-html

venv:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install numpy scipy matplotlib nbformat nbconvert jupyter ipykernel

kernel:
	.venv/Scripts/python -m ipykernel install --user --name ml-scratchpad --display-name "ML Scratchpad (shared)"

install:
	pip install numpy scipy matplotlib nbformat nbconvert jupyter ipykernel

# ── Open a subfolder's teaching notebook ──────────────────────────────────────
turbo-quant-notebook:
	jupyter notebook turbo-quant/turbo_quant_explainer.ipynb

attention-transformer-notebook:
	jupyter notebook attention-transformer/attention_explainer.ipynb

nanogpt-notebook:
	jupyter notebook nanogpt-from-scratch/nanogpt_explainer.ipynb

kv-cache-notebook:
	jupyter notebook kv-cache/kvcache_explainer.ipynb

rope-notebook:
	jupyter notebook rope/rope_explainer.ipynb

inference-engine-notebook:
	jupyter notebook inference-engine/inference_engine_explainer.ipynb

continuous-batching-notebook:
	jupyter notebook continuous-batching/continuous_batching_explainer.ipynb

vllm-memory-notebook:
	jupyter notebook vllm-memory/vllm_memory_explainer.ipynb

additive-inference-notebook:
	jupyter notebook additive-inference/additive_inference_explainer.ipynb

# ── Maintenance (Claude regenerates notebooks from generators) ─────────────────
turbo-quant-regen:
	cd turbo-quant && python create_notebook.py

attention-transformer-regen:
	cd attention-transformer && python create_notebook.py

nanogpt-regen:
	cd nanogpt-from-scratch && python create_notebook.py

kv-cache-regen:
	cd kv-cache && python create_notebook.py

rope-regen:
	cd rope && python create_notebook.py

inference-engine-regen:
	cd inference-engine && python create_notebook.py

continuous-batching-regen:
	cd continuous-batching && python create_notebook.py

vllm-memory-regen:
	cd vllm-memory && python create_notebook.py

# Publish: self-contained HTML of the executed notebook, served by GitHub Pages at
# https://ylnhari.github.io/ml-scratchpad/vllm-memory/ . Re-run after ANY change to the notebook.
vllm-memory-html:
	jupyter nbconvert --to html --embed-images vllm-memory/vllm_memory_explainer.ipynb --output index.html --output-dir vllm-memory
	python -c "import io; p='vllm-memory/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>vllm_memory_explainer</title>', '<title>vLLM GPU Memory — From Startup Profiling to Serving Requests</title>'))"

additive-inference-regen:
	cd additive-inference && python create_notebook.py

# Publish the remaining 8 subfolders the same way — self-contained HTML of the
# executed notebook, served by GitHub Pages at https://ylnhari.github.io/ml-scratchpad/<folder>/.
# Re-run after ANY change to the notebook.
turbo-quant-html:
	jupyter nbconvert --to html --embed-images turbo-quant/turbo_quant_explainer.ipynb --output index.html --output-dir turbo-quant
	python -c "import io; p='turbo-quant/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>turbo_quant_explainer</title>', '<title>TurboQuant — Step-by-Step Visual Guide</title>'))"

attention-transformer-html:
	jupyter nbconvert --to html --embed-images attention-transformer/attention_explainer.ipynb --output index.html --output-dir attention-transformer
	python -c "import io; p='attention-transformer/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>attention_explainer</title>', '<title>Attention Is All You Need — Step-by-Step Visual Guide</title>'))"

nanogpt-html:
	jupyter nbconvert --to html --embed-images nanogpt-from-scratch/nanogpt_explainer.ipynb --output index.html --output-dir nanogpt-from-scratch
	python -c "import io; p='nanogpt-from-scratch/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>nanogpt_explainer</title>', '<title>nanoGPT From Scratch — Watch a Transformer Learn</title>'))"

kv-cache-html:
	jupyter nbconvert --to html --embed-images kv-cache/kvcache_explainer.ipynb --output index.html --output-dir kv-cache
	python -c "import io; p='kv-cache/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>kvcache_explainer</title>', '<title>The KV-Cache — Make Generation Fast Without Changing the Answer</title>'))"

rope-html:
	jupyter nbconvert --to html --embed-images rope/rope_explainer.ipynb --output index.html --output-dir rope
	python -c "import io; p='rope/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>rope_explainer</title>', '<title>RoPE — Rotary Positional Embeddings From Scratch</title>'))"

inference-engine-html:
	jupyter nbconvert --to html --embed-images inference-engine/inference_engine_explainer.ipynb --output index.html --output-dir inference-engine
	python -c "import io; p='inference-engine/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>inference_engine_explainer</title>', '<title>A Tiny Inference Engine — Paged KV-Cache + Speculative Decoding</title>'))"

continuous-batching-html:
	jupyter nbconvert --to html --embed-images continuous-batching/continuous_batching_explainer.ipynb --output index.html --output-dir continuous-batching
	python -c "import io; p='continuous-batching/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>continuous_batching_explainer</title>', '<title>Continuous Batching — Keep the Batch Full</title>'))"

additive-inference-html:
	jupyter nbconvert --to html --embed-images additive-inference/additive_inference_explainer.ipynb --output index.html --output-dir additive-inference
	python -c "import io; p='additive-inference/index.html'; s=io.open(p,encoding='utf-8').read(); io.open(p,'w',encoding='utf-8').write(s.replace('<title>additive_inference_explainer</title>', '<title>Inference Without Multiplying: How a 1.7-bit Model Works</title>'))"

# ── Housekeeping ──────────────────────────────────────────────────────────────
deps:
	@echo "Subfolder dependency groups in pyproject.toml:"
	@python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); \
	  [print(f'  [{k}]') for k in d['project']['optional-dependencies'] if k != 'all']"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "Clean."
