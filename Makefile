.PHONY: install test test-domain test-fast test-slow lint type clean \
	scrape scrape-run scrape-smoke scrape-side scrape-debug \
	build-dataset inspect

# Optional CLI args passthrough for scrape-run.
# Example:
#   make scrape-run SCRAPE_ARGS='--query-contains "side view" --max-queries 10 --max-results 80'
SCRAPE_ARGS ?=

install:
	poetry install

test:
	poetry run pytest

test-domain:
	poetry run pytest tests/domain -v

test-fast:
	poetry run pytest -m "not slow" -v

test-slow:
	poetry run pytest -m slow -v

lint:
	poetry run ruff check sdi_helper tests

type:
	poetry run mypy sdi_helper

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

scrape:
	poetry run sdi-helper

scrape-run:
	poetry run sdi-helper $(SCRAPE_ARGS)

scrape-smoke:
	poetry run sdi-helper --max-queries 1 --max-results 10 --verbose

scrape-side:
	poetry run sdi-helper --query-contains "side view" --max-queries 10 --max-results 80

scrape-debug:
	poetry run sdi-helper --query-contains "side view" --max-queries 5 --max-results 40 --verbose

build-dataset:
	poetry run python -m sdi_helper.interfaces.cli.build_dataset

inspect:
	poetry run python -m sdi_helper.interfaces.cli.inspect_state
