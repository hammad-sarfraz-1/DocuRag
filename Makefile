# DDRPRIV — agentic RAG chatbot
# The frontend and backend are ONE Uvicorn process (FastAPI serves the API and
# the static HTML), so `make up` / `make down` bring both online/offline together.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
APP     := backend.app:app
HOST    := 0.0.0.0
PORT    ?= 8000
PIDFILE := .server.pid
LOG     := server.log

.DEFAULT_GOAL := help
.PHONY: help install up down restart status logs clean

help:               ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install:            ## Create the venv and install dependencies
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt
	@echo "Dependencies installed. Run 'make up'."

up:                 ## Start frontend + backend on $(PORT)
	@if [ ! -x "$(UVICORN)" ]; then echo "venv missing — run 'make install' first"; exit 1; fi; \
	if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "Already running (PID $$(cat $(PIDFILE))) -> http://localhost:$(PORT)"; exit 0; \
	fi; \
	echo "Starting http://localhost:$(PORT) (first boot downloads models, ~20s) ..."; \
	nohup $(UVICORN) $(APP) --host $(HOST) --port $(PORT) > $(LOG) 2>&1 & echo $$! > $(PIDFILE); \
	for i in $$(seq 1 90); do \
		if curl -sf http://localhost:$(PORT)/health >/dev/null 2>&1; then \
			echo "Ready -> http://localhost:$(PORT)  (PID $$(cat $(PIDFILE)), logs: make logs)"; exit 0; \
		fi; \
		if ! kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
			echo "Failed to start — last log lines:"; tail -n 15 $(LOG); rm -f $(PIDFILE); exit 1; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for /health — check 'make logs'"; exit 1

down:               ## Stop frontend + backend
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		kill $$(cat $(PIDFILE)) && echo "Stopped (PID $$(cat $(PIDFILE)))."; \
		rm -f $(PIDFILE); \
	elif command -v fuser >/dev/null 2>&1 && fuser $(PORT)/tcp >/dev/null 2>&1; then \
		fuser -k $(PORT)/tcp >/dev/null 2>&1 && echo "Stopped via port $(PORT)."; \
		rm -f $(PIDFILE); \
	else \
		echo "Not running."; rm -f $(PIDFILE); \
	fi

restart:            ## Restart frontend + backend
	@$(MAKE) --no-print-directory down
	@sleep 1
	@$(MAKE) --no-print-directory up

status:             ## Show whether the app is running
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		printf "Running (PID %s) — health: " "$$(cat $(PIDFILE))"; \
		curl -s http://localhost:$(PORT)/health || echo "(no response)"; echo; \
	else \
		echo "Not running."; \
	fi

logs:               ## Follow the server log
	@touch $(LOG); tail -n 50 -f $(LOG)

clean:              ## Stop the app and remove pidfile + log
	@$(MAKE) --no-print-directory down >/dev/null 2>&1 || true
	@rm -f $(PIDFILE) $(LOG)
	@echo "Cleaned."
