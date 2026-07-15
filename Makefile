# DDRPRIV — agentic RAG chatbot
# The frontend and backend are ONE Uvicorn process (FastAPI serves the API and
# the static HTML), so `make up` / `make down` bring both online/offline together.

IMAGE       := docurag
CONTAINER   := docurag_dev
PORT        ?= 8000
PERSIST_DIR := $(CURDIR)/chroma_db
LOGS_DIR    := $(CURDIR)/logs
CHAT_META_DIR := $(CURDIR)/chat_meta

.DEFAULT_GOAL := help
.PHONY: help build up down restart status logs clean

help:               ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

build:              ## Build the Docker image
	docker build -t $(IMAGE) .

up:                 ## Start frontend + backend on $(PORT) (Docker) — run 'make build' first
	@mkdir -p $(PERSIST_DIR) $(LOGS_DIR) $(CHAT_META_DIR)
	@test -f $(CHAT_META_DIR)/chat_metadata.json || echo '{}' > $(CHAT_META_DIR)/chat_metadata.json
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 || true
	@echo "Starting http://localhost:$(PORT) (first boot downloads models, ~20s) ..."
	@docker run -d --name $(CONTAINER) \
		--env-file .env \
		-p $(PORT):8000 \
		-v $(PERSIST_DIR):/app/chroma_db \
		-v $(LOGS_DIR):/app/logs \
		-v $(CHAT_META_DIR):/app/chat_meta \
		-e CHAT_META_FILE=/app/chat_meta/chat_metadata.json \
		$(IMAGE) >/dev/null
	@for i in $$(seq 1 90); do \
		if curl -sf http://localhost:$(PORT)/health >/dev/null 2>&1; then \
			echo "Ready -> http://localhost:$(PORT)  (container: $(CONTAINER), logs: make logs)"; exit 0; \
		fi; \
		if ! docker ps -q -f name=^/$(CONTAINER)$$ | grep -q .; then \
			echo "Failed to start — last log lines:"; docker logs --tail 30 $(CONTAINER); exit 1; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for /health — check 'make logs'"; exit 1

down:               ## Stop frontend + backend
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 && echo "Stopped." || echo "Not running."

restart:            ## Restart frontend + backend
	@$(MAKE) --no-print-directory down
	@$(MAKE) --no-print-directory up

status:             ## Show whether the app is running
	@if docker ps -q -f name=^/$(CONTAINER)$$ | grep -q .; then \
		printf "Running (container: $(CONTAINER)) — health: "; \
		curl -s http://localhost:$(PORT)/health || echo "(no response)"; echo; \
	else \
		echo "Not running."; \
	fi

logs:               ## Follow the server log
	@docker logs -f --tail 50 $(CONTAINER)

clean:              ## Stop the app and remove the container
	@$(MAKE) --no-print-directory down >/dev/null 2>&1 || true
	@echo "Cleaned."
