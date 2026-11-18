.PHONY: install test data bench report app manual
install: ; uv sync --extra speaker --extra app --group dev
test: ; uv run pytest -q
data: ; uv run voicetrap build
bench: ; uv run voicetrap bench --run main
report: ; uv run voicetrap report --run main
app: ; uv run streamlit run app.py
manual: ; python3 docs/manual/build.py
