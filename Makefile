PYTHON ?= python3
CHECK_DIRS=src

.PHONY: all_tests test_install pylint pytest

all_tests: test_install
	$(MAKE) pylint pytest

test_install:
	$(PYTHON) -m pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements/devel.txt scikit-learn==1.7.2 scipy==1.14.1 tqdm==4.67.1
	$(PYTHON) -m pip install .

pylint:
	$(PYTHON) -m pylint $(CHECK_DIRS)/

# Resolve the installed path without importing the package during coverage setup.
pytest:
	$(PYTHON) -m pytest tests/ -v --cov=$(CHECK_DIRS) \
		--cov="$$($(PYTHON) -I -c 'from importlib.metadata import distribution; print(distribution("kaiwu-torch-plugin").locate_file("kaiwu/torch_plugin"))')" \
		--cov-report=term
