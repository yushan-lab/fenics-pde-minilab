PYTHON ?= python
IMAGE ?= fenics-pde-minilab

.PHONY: setup poisson heat reproduce test clean-results docker-build docker-reproduce

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

poisson:
	$(PYTHON) scripts/run_poisson.py

heat:
	$(PYTHON) scripts/run_heat.py

reproduce:
	$(PYTHON) scripts/reproduce_all.py

test:
	$(PYTHON) -m pytest -q

clean-results:
	$(PYTHON) scripts/clean_results.py

docker-build:
	docker build -t $(IMAGE) .

docker-reproduce: docker-build
	docker run --rm -v "$(CURDIR):/workspace" -w /workspace $(IMAGE) make reproduce
