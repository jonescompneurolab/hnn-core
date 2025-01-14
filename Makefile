# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules

.PHONY: all modl clean check-manifest

all: modl

modl:
	cd hnn_core/mod/ && nrnivmodl

clean :
	rm -rf hnn_core/mod/x86_64/*
	rm -rf hnn_core/mod/arm64/*

check-manifest:
	check-manifest

test: lint
	pytest ./hnn_core/tests/ -m "not uses_mpi" -n auto
	pytest ./hnn_core/tests/ -m "uses_mpi"

lint:
	@if command -v ruff > /dev/null; then \
		echo "Running ruff check"; \
		ruff check hnn_core; \
	else \
		echo "ruff not found, please install it!"; \
		exit 1; \
	fi;
