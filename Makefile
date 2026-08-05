# ── Shared environment ────────────────────────────────────────────────────────
# One .venv at learning/ root. Run `make venv` once, then `make kernel`.

.PHONY: venv install kernel deps clean nanogpt-notebook nanogpt-regen kv-cache-notebook kv-cache-regen rope-notebook rope-regen inference-engine-notebook inference-engine-regen continuous-batching-notebook continuous-batching-regen turbo-quant-notebook attention-transformer-notebook turbo-quant-regen attention-transformer-regen additive-inference-notebook additive-inference-regen

venv:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install numpy scipy matplotlib nbformat nbconvert jupyter ipykernel

kernel:
	.venv/Scripts/python -m ipykernel install --user --name learning --display-name "Learning (shared)"

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

additive-inference-regen:
	cd additive-inference && python create_notebook.py

# ── Housekeeping ──────────────────────────────────────────────────────────────
deps:
	@echo "Subfolder dependency groups in pyproject.toml:"
	@python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); \
	  [print(f'  [{k}]') for k in d['project']['optional-dependencies'] if k != 'all']"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "Clean."
