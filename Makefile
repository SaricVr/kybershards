.PHONY: lint lint-mix lint-all test test-all

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run pyright

lint-mix:
	uv run tombi format pyproject.toml
	uv run tombi lint pyproject.toml
	uv run yamllint .

lint-all: lint lint-mix

test:
	uv run pytest --cov=kybershards --cov-branch --cov-report=term-missing

test-all:
	uv run coverage erase
	uv run tox -p auto
	uv run coverage report
