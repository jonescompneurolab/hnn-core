# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules
all: modl

modl:
	cd hnn_core/mod/ && nrnivmodl

# clean
.PHONY: clean
clean :
	rm -rf hnn_core/mod/x86_64/*
