CHECK_DIRS=src

all_tests: test_install
	$(MAKE) pylint
	$(MAKE) pytest

test_install:
	pip3 install -i https://mirrors.aliyun.com/pypi/simple -r requirements/devel.txt

pylint:
	pylint src/

pytest:
	coverage run --source=$(CHECK_DIRS) -m pytest tests --ignore=tests/
	coverage report
