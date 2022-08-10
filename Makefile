# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules

.PHONY: all modl clean check-manifest

all: modl

modl:
	cd hnn_core/mod/ && nrnivmodl

clean :
	rm -rf hnn_core/mod/x86_64/*

check-manifest:
	check-manifest

test: flake
	pytest .

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 hnn_core --count; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"
