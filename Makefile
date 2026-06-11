# APGI Implementation — Makefile
# Reproduces all figures referenced in Paper 1 of the APGI series.
#
# Usage:
#   make reproduce-paper   — generate all paper figures into outputs/
#   make figure-s1         — Figure S1 only (onset timeline schematic)
#   make figure-07         — Hierarchical system plots
#   make figure-08         — Spectral validation plots
#   make test              — run full test suite
#   make clean             — remove pycache and temporary files
#   make clean-outputs     — remove generated outputs

PYTHON      ?= python3
OUTPUT_DIR   = outputs
EXAMPLES_DIR = examples

.PHONY: reproduce-paper figure-s1 figure-07 figure-08 test clean clean-outputs

## Primary target: reproduce all paper figures
reproduce-paper: $(OUTPUT_DIR)
	@echo "================================================================"
	@echo "Reproducing APGI Paper 1 figures"
	@echo "Output directory: $(OUTPUT_DIR)/"
	@echo "================================================================"
	$(PYTHON) $(EXAMPLES_DIR)/figure_s1.py
	$(PYTHON) $(EXAMPLES_DIR)/07_hierarchical_system.py
	$(PYTHON) $(EXAMPLES_DIR)/08_spectral_validation.py
	@echo "================================================================"
	@echo "All figures written to $(OUTPUT_DIR)/"
	@echo "================================================================"

## Individual figure targets
figure-s1: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/figure_s1.py

figure-07: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/07_hierarchical_system.py

figure-08: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/08_spectral_validation.py

## Test suite
test:
	$(PYTHON) -m pytest tests/ -v

## Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

## Clean generated outputs
clean-outputs:
	rm -rf $(OUTPUT_DIR)

## Clean pycache and temporary files
clean:
	$(PYTHON) delete_pycache.py --yes
