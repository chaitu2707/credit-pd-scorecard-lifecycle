.PHONY: data features train validate monitor app test all

data:
	python -m src.data_prep

features:
	@echo "Stage 3: feature selection (notebook 03). Not yet automated."

train:
	@echo "Stage 4-5: scorecard and challengers (notebooks 04-05). Not yet automated."

validate:
	@echo "Stage 6: validation (notebook 06). Not yet automated."

monitor:
	@echo "Stage 7: monitoring (notebook 07). Not yet automated."

app:
	streamlit run app/Home.py

test:
	python -m pytest -q

all: data test
