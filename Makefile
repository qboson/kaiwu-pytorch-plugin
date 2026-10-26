PYTHON ?= python3
CHECK_DIRS=src

.PHONY: all_tests test_install pylint pytest

all_tests: test_install
	$(MAKE) pylint pytest

test_install:
	$(PYTHON) -m pip install -r requirements/devel.txt
	$(PYTHON) -m pip install .

pylint:
	$(PYTHON) -m pylint $(CHECK_DIRS)/

# Resolve the installed path without importing the package during coverage setup.
pytest:
	$(PYTHON) -m pytest tests/ -v --cov=$(CHECK_DIRS) \
		--cov="$$($(PYTHON) -I -c 'from importlib.metadata import distribution; print(distribution("kaiwu-torch-plugin").locate_file("kaiwu/torch_plugin"))')" \
		--cov-report=term
