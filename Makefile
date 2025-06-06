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
	python -m unittest discover -v

# Rule to check the coverage of the package tests
coverage:
	coverage run -m unittest discover -v
	coverage report --fail-under=90

# Rule to build the package the same way as it would be built for distribution
build:
	python -m build

dev-dependencies:
	pip install -e ".[dev]"

hooks:
	pre-commit install
	pre-commit run --all-files

.PHONY: docs-help generate-docs tests coverage Makefile
