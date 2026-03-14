# Plan: Chatbot & LLM-Advisor Statistics

## Goal
Transition the LLM-Advisor from using local artifacts and backtest data to a fully automated, live-tracking telemetry system that feeds both the project dashboard and the chatbot's context.

## Phase 1: Infrastructure Activation & Automation
1. **Apply Migrations:** Run `005_llm_advisor_telemetry.sql` against your Supabase instance to create the necessary tables (`runs`, `trades`, `heartbeats`).
2. **Environment Configuration:** Ensure `.env` is populated with `LLM_ADVISOR_CRON_SECRET` and `LLM_ADVISOR_DAILY_NEWS_DIR`.
3. **Automate Ingestion:** Instead of manually running the `curl` command, set up a cron job (using Vercel Cron, GitHub Actions, or a standard server cron) to hit `POST /api/llm-advisor/metrics` on a schedule to keep the database fresh.

## Phase 2: Transition to Live Broker Data
1. **Broker API Integration:** Connect to your live broker's API (e.g., Alpaca, Interactive Brokers) to fetch real fill and execution logs.
2. **Schema Update:** Either add an `environment` column (e.g., `backtest` vs `live`) to your existing tables or create new tables specifically for live trades.
3. **Live Ingestion Route:** Create a new endpoint or update the existing one to ingest real trade data into Supabase so P/L and Win Rate reflect actual money, not just backtests.

## Phase 3: Frontend & Chatbot Wiring
1. **Dashboard Toggle:** Update the `/projects/llm-advisor` page to allow users to toggle between "Backtest Performance" and "Live Performance".
2. **Chatbot Context Awareness:** Give your chatbot read-access to the `GET /api/llm-advisor/metrics` endpoint. If a user asks the chatbot "How are you performing this week?", the chatbot should query its own live telemetry to provide an accurate, data-backed answer.
