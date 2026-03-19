.PHONY: help sync activate test run

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync      - Create/update the local uv environment with dev dependencies' \
		'  make activate  - Print the command to activate the local virtual environment' \
		'  make test      - Run the tokenizer unit tests inside uv' \
		'  make run       - Run the training script inside uv'

sync:
	uv sync --dev

activate:
	@printf '%s\n' 'Run this command in your shell:'
	@printf '%s\n' 'source .venv/bin/activate'

test:
	uv run python -m unittest test_train_tokenizer.py

run:
	uv run python main.py
