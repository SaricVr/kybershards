.PHONY: lint lint-mix lint-all tests test-all docs

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run pyright

lint-mix:
	uv run tombi format pyproject.toml
	uv run tombi lint pyproject.toml
	uv run yamllint .
	uv run pymarkdownlnt fix docs *.md -r
	uv run pymarkdownlnt scan docs *.md -r

lint-all: lint lint-mix

test:
	uv run pytest --cov=kybershards --cov-branch --cov-report=term-missing -v -n auto

test-all:
	uv run --group test coverage erase
	uv run tox -p auto
	uv run coverage report

docs:
	uv run --group docs mkdocs build --strict
