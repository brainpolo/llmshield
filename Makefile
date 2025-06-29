# You can set these variables from the command line, and also
# from the environment for the first two.

# Options for the Sphinx build executable
SPHINXOPTS    ?=

# Sphinx build executable
SPHINXBUILD   ?= sphinx-build

# Source directory for the documentation. Must contain the conf.py file which determines the configuration of the documentation build.
SOURCEDIR     = docs/source

# Build directory for the documentation. This is where the generated files will be placed.
BUILDDIR      = docs/out

# Rule to display Sphinx help
docs-help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Rule to generate documentation in HTML format
generate-docs:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Rule to run all of the package tests
tests:
	python3.12 -m unittest discover -v

# Rule to check the coverage of the package tests
coverage:
	coverage run -m unittest discover -v
	coverage report

# Rule to build the package the same way as it would be built for distribution
build:
	python -m build

# Rule to verify package builds correctly and all resources are accessible
verify-package:
	python3.12 -m unittest tests.test_package_installation -v

dev-dependencies:
	pip install -e ".[dev]"

hooks:
	pre-commit install
	pre-commit run --all-files

# Rule to check documentation coverage using Ruff
doc-coverage:
	@echo "Checking docstring coverage with Ruff..."
	@echo "Files checked: $$(find llmshield -name '*.py' -not -path '*/tests/*' | wc -l)"
	@echo "Docstring issues found:"
	@ruff check llmshield/ --statistics || true

# Rule to run Ruff linting and formatting
ruff:
	ruff check llmshield/ tests/ --fix
	ruff format llmshield/ tests/

# Rule to run Ruff check only (no fixes)
ruff-check:
	ruff check llmshield/ tests/
	ruff format llmshield/ tests/ --check

.PHONY: docs-help generate-docs tests coverage doc-coverage ruff ruff-check verify-package Makefile
