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
