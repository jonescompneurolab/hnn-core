# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules

.PHONY: all modl clean check-manifest

all: modl

clean :
	rm -rf hnn_core/mod/x86_64/*

check-manifest:
	check-manifest

format:
	ruff format

lint:
	ruff check

modl:
	cd hnn_core/mod/ && nrnivmodl

test:
	ruff format --check
	ruff check --no-fix
	pytest .
