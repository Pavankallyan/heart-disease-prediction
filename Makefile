.PHONY: setup test summary

setup:
	pip install -r requirements.txt

test:
	python3 -m pytest tests/ -q

summary:
	python3 examples/data_summary.py
