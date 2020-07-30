# Makefile for model - compiles mod files for use by NEURON
# first rev: (SL: created)

# make rules
all: modl

modl:
	cd $(PWD)/mod/ && nrnivmodl

# clean
.PHONY: clean
clean :
	rm -f hnn_core/mod/x86_64/*
