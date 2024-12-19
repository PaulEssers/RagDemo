start-llm:
	docker compose build
	docker compose up -d

test-llm:
	curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d '{"text": "Hello!"}'

run: start-llm
	python src/ui.py