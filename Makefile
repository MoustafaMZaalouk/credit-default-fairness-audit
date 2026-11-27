export PYTHONPATH := src

.PHONY: test smoke run lint
test:
	python -m pytest -q
smoke:
	python -m creditaudit.run --subsample 2000 --seeds 1 --out /tmp/credit-smoke
run:
	python -m creditaudit.run
lint:
	ruff check .
