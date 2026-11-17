.PHONY: install test data bench report
install: ; uv sync --extra speaker --group dev
test: ; uv run pytest -q
data: ; uv run voicetrap build
bench: ; uv run voicetrap bench --run main
report: ; uv run voicetrap report --run main
