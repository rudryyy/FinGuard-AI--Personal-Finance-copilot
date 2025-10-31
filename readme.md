💸 FinGuard AI — Personal Finance Copilot

A Streamlit-powered personal finance assistant that helps you chat with your spending data, get live market prices, and summarize monthly budgets using AI models (Gemini / Groq / Ollama).

⚠️ This app is for educational purposes only — not investment or tax advice.

🚀 Features

💬 AI Chatbot Interface — Ask finance-related questions in natural language.

🧾 Smart Transaction Parsing — Converts everyday text like “Paid ₹250 on Zomato for lunch yesterday” into structured records.

📊 Monthly Reports — Visualize spending by category and receive personalized summaries.

💹 Live Market Data — Fetch near-real-time NSE/BSE quotes via Twelve Data (or Yahoo Finance fallback).

🔄 Multiple LLM Providers — Switch between:

Ollama (local) — Run Llama-3 locally.

Gemini (Google AI) — Fast, reliable cloud model.

Groq (cloud) — Llama-3 via Groq API.

🗄️ Local Database — Stores transactions in a lightweight SQLite finance.db.

🧰 Tech Stack

Frontend / UI: Streamlit

Backend: Python 3.11 + SQLite

AI Providers: Ollama, Gemini, Groq

Market Data: Twelve Data API, Yahoo Finance (yfinance)

Visualization: Matplotlib + Pandas
