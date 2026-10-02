# Thin wrapper so `make <task>` works where make is installed (Linux, macOS).
# On Windows, run `python run.py <task>` instead; run.py is the source of truth.
PYTHON ?= python
TASKS := test check-env generator-report set-b-report data baselines train-evaluator train-cvae evaluator-report generate

.PHONY: $(TASKS)
$(TASKS):
	$(PYTHON) run.py $@
