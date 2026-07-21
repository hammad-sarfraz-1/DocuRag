# DDRPRIV — agentic RAG chatbot
# The frontend and backend are ONE Uvicorn process (FastAPI serves the API and
# the static HTML); Postgres runs alongside it via docker-compose.

IMAGE       := docurag
PORT        ?= 8000
PERSIST_DIR := $(CURDIR)/chroma_db
LOGS_DIR    := $(CURDIR)/logs
PGDATA_DIR  := $(CURDIR)/pgdata

.DEFAULT_GOAL := help
.PHONY: help build up down restart status logs clean

help:               ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

build:              ## Build the Docker image
	docker build -t $(IMAGE) .

up:                 ## Start frontend + backend + postgres (Docker Compose) — run 'make build' first
	@mkdir -p $(PERSIST_DIR) $(LOGS_DIR) $(PGDATA_DIR)
	@PORT=$(PORT) docker compose up -d
	@for i in $$(seq 1 90); do \
		if curl -sf http://localhost:$(PORT)/health >/dev/null 2>&1; then \
			echo "Ready -> http://localhost:$(PORT)  (logs: make logs)"; exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for /health — check 'make logs'"; exit 1

down:               ## Stop frontend + backend + postgres
	docker compose down

restart:            ## Restart frontend + backend + postgres
	@$(MAKE) --no-print-directory down
	@$(MAKE) --no-print-directory up

status:             ## Show whether the app is running
	@docker compose ps
	@printf "health: "; curl -s http://localhost:$(PORT)/health || echo "(no response)"; echo

logs:               ## Follow the server log
	docker compose logs -f --tail 50 app

clean:              ## Stop the app and wipe chroma_db/, logs, and the postgres volume
	@$(MAKE) --no-print-directory down >/dev/null 2>&1 || true
	@docker run --rm \
		-v $(PERSIST_DIR):/target/chroma_db \
		-v $(LOGS_DIR):/target/logs \
		-v $(PGDATA_DIR):/target/pgdata \
		alpine sh -c 'rm -rf /target/chroma_db/* /target/chroma_db/.[!.]* /target/logs/* /target/logs/.[!.]* /target/pgdata/* /target/pgdata/.[!.]*' 2>/dev/null || true
	@echo "Cleaned."
