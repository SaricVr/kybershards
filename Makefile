.PHONY: lint test lint-mix lint-all

lint:
	uv run ruff format
	uv run ruff check --fix

lint-mix:
	uv run tombi format pyproject.toml
	uv run tombi lint pyproject.toml
	uv run yamllint .

test:
	uv run pytest --cov=kybershards --cov-branch --cov-report=term-missing

lint-all: lint lint-mix