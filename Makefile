# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules

.PHONY: all clean check-manifest lint modl regenerate-test-network test

all: modl

clean :
	rm -rf hnn_core/mod/x86_64
	rm -rf hnn_core/mod/arm64

check-manifest:
	check-manifest

format-check:
	@if command -v ruff > /dev/null; then \
		echo "Running check of ruff format"; \
		ruff format hnn_core --check; \
	else \
		echo "ruff not found, please install it!"; \
		exit 1; \
	fi;

format-overwrite:
	@if command -v ruff > /dev/null; then \
		echo "Running ruff format, this will likely change some code!"; \
		ruff format hnn_core; \
	else \
		echo "ruff not found, please install it!"; \
		exit 1; \
	fi;

lint:
	@if command -v ruff > /dev/null; then \
		echo "Running ruff check"; \
		ruff check hnn_core; \
	else \
		echo "ruff not found, please install it!"; \
		exit 1; \
	fi;

modl:
	cd hnn_core/mod/ && nrnivmodl

regenerate-networks:
	python ./dev_scripts/regenerate_base_network.py
	python ./hnn_core/tests/regenerate_test_network.py

spell:
	@if command -v codespell > /dev/null; then \
		echo "Running codespell"; \
		codespell; \
	else \
		echo "codespell not found, please install it!"; \
		exit 1; \
	fi;

test: format-check lint spell
	pytest ./hnn_core/tests/ -m "not uses_mpi" -n auto
	pytest ./hnn_core/tests/ -m "uses_mpi"
