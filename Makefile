.PHONY: install test doctor plan smoke clean ood aggregate

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

doctor:
	python -m fastwam_ood_eval.cli doctor --config configs/eval_clean_smoke.yaml

plan:
	python -m fastwam_ood_eval.cli plan --config configs/eval_ood_smoke.yaml

smoke:
	bash scripts/run_smoke_test.sh

clean:
	bash scripts/run_clean_eval.sh

ood:
	bash scripts/run_ood_eval.sh

aggregate:
	bash scripts/aggregate_results.sh

