# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules

.PHONY: all clean check-manifest lint modl regenerate-test-network test

all: modl

clean :
	rm -rf hnn_core/mod/x86_64/*
	rm -rf hnn_core/mod/arm64/*

check-manifest:
	check-manifest

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

regenerate-test-network:
	python ./hnn_core/tests/regenerate_test_network.py

spell:
	@if command -v codespell > /dev/null; then \
		echo "Running codespell"; \
		codespell; \
	else \
		echo "codespell not found, please install it!"; \
		exit 1; \
	fi;

test: lint spell
	pytest ./hnn_core/tests/ -m "not uses_mpi" -n auto
	pytest ./hnn_core/tests/ -m "uses_mpi"
