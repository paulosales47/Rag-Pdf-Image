.PHONY: install test lint format ingest ask app up down restart status clean-db logs

install:
	pip install -e ".[dev,app]"
	pre-commit install

app:
	streamlit run src/rag_pdf/app/streamlit_app.py

up:
	scripts/rag-pdf.sh start

down:
	scripts/rag-pdf.sh stop

restart:
	scripts/rag-pdf.sh restart

status:
	scripts/rag-pdf.sh status

clean-db:
	scripts/rag-pdf.sh clean

logs:
	scripts/rag-pdf.sh logs

test:
	pytest --cov=rag_pdf

lint:
	ruff check .
	mypy src

format:
	ruff format .

ingest:
	rag-pdf ingest --pdf $(PDF)

ask:
	rag-pdf ask "$(Q)"
