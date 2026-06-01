PYTHON := .venv/bin/python
STREAMLIT := .venv/bin/streamlit

.PHONY: help install pipeline bronze silver gold models ingest ingest-check app clean

help:
	@echo "Targets:"
	@echo "  make install       - crea .venv e instala dependencias"
	@echo "  make pipeline      - corre bronze -> silver -> gold -> models"
	@echo "  make bronze        - solo bronze"
	@echo "  make silver        - solo silver"
	@echo "  make gold          - solo gold"
	@echo "  make models        - K-Means + FP-Growth + ALS (segmentación y recomendador)"
	@echo "  make ingest-check  - reporta qué archivos en data/landing/ son nuevos / cambiaron"
	@echo "  make ingest        - ejecuta el pipeline completo si hay datos nuevos"
	@echo "  make app           - levanta el dashboard Streamlit en :8501"
	@echo "  make clean         - borra data/{bronze,silver,gold,models} y caches"

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) -m src.pipeline.run --step all

bronze:
	$(PYTHON) -m src.pipeline.run --step bronze

silver:
	$(PYTHON) -m src.pipeline.run --step silver

gold:
	$(PYTHON) -m src.pipeline.run --step gold

models:
	$(PYTHON) -m src.pipeline.run --step models

ingest-check:
	$(PYTHON) -m src.pipeline.ingest --check

ingest:
	$(PYTHON) -m src.pipeline.ingest --run

app:
	$(STREAMLIT) run app/streamlit_app.py

clean:
	rm -rf data/bronze data/silver data/gold data/models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
