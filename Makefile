all:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository testpypi dist/*

setup:
	python3 -m pip install --user --upgrade setuptools wheel
	python3 -m pip install --user --upgrade twine
