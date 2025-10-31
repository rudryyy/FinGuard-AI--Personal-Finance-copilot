# app.py — Personal Finance Chatbot (Ollama + Gemini + Groq) with robust live quotes
# -----------------------------------------------------------------------------------
# - Switch LLM provider in sidebar: Ollama (local), Gemini (Google AI), or Groq (cloud)
# - Live quotes: Twelve Data (near-real-time) or Yahoo/yfinance (free, delayed)
# - Quotes include as_of + age_minutes, always injected into the LLM prompt
# - Transactions: NL → structured via LLM, CSV import, monthly reports

from __future__ import annotations
import os, re, json, sqlite3, datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import requests

# ---------------------------- Streamlit setup ---------------------------------
st.set_page_config(page_title="Personal Finance Chatbot — Llama 3", layout="wide")

DB_PATH = "finance.db"
SYSTEM_PROMPT = (
    "You are a personal finance assistant for an Indian user.\n"
    "- Be short, practical, and non-judgmental.\n"
    "- Assume INR unless specified.\n"
    "- Provide general education only (no tax/investment advice).\n"
)

# ---------------------- Built-in name → ticker mappings -----------------------
NAME_TO_TICKER = {
    # Common examples
    "reliance": "RELIANCE.NS", "ril": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "hdfc": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS", "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "lt": "LT.NS", "larsen": "LT.NS",
    "airtel": "BHARTIARTL.NS", "bharti airtel": "BHARTIARTL.NS",
    "bharati airtel": "BHARTIARTL.NS", "bharati": "BHARTIARTL.NS", "bharti": "BHARTIARTL.NS",
    "asian paints": "ASIANPAINT.NS",
    "itc": "ITC.NS",
    "maruti": "MARUTI.NS",

    # Adani Green
    "adani green": "ADANIGREEN.NS",
    "adani green energy": "ADANIGREEN.NS",
    "adani green energy ltd": "ADANIGREEN.NS",

    # NTPC / NTPC Green
    "ntpc": "NTPC.NS",
    "ntpc green": "NTPCGREEN.NS",
    "ntpc green energy": "NTPCGREEN.NS",
}

# ------------------------------ Lazy imports ----------------------------------
def lazy_import_ollama():
    try:
        import ollama
        return ollama, None
    except Exception as e:
        return None, e

def lazy_import_groq():
    try:
        from groq import Groq
        return Groq, None
    except Exception as e:
        return None, e

def lazy_import_gemini():
    try:
        import google.generativeai as genai
        return genai, None
    except Exception as e:
        return None, e

def lazy_import_yfinance():
    try:
        import yfinance as yf
        return yf, None
    except Exception as e:
        return None, e

def lazy_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt, None
    except Exception as e:
        return None, e

# --------------------------------- Timezone -----------------------------------
try:
    from zoneinfo import ZoneInfo
    INDIA_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    INDIA_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))

# ---------------------------------- DB ----------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tdate TEXT,
            amount REAL,
            currency TEXT,
            merchant TEXT,
            description TEXT,
            ttype TEXT,
            category TEXT,
            source TEXT
        )
        """)
        conn.commit()
    finally:
        conn.close()

def insert_txn(row: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO transactions (tdate, amount, currency, merchant, description, ttype, category, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("tdate"),
        float(row.get("amount", 0) or 0),
        row.get("currency", "INR"),
        row.get("merchant"),
        row.get("description"),
        row.get("ttype"),
        row.get("category"),
        row.get("source", "manual"),
    ))
    conn.commit()
    conn.close()

def read_txns() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY tdate DESC, id DESC", conn)
    conn.close()
    return df

# --------------------------------- Model --------------------------------------
@dataclass
class FinanceItem:
    tdate: str
    amount: float
    currency: str
    merchant: Optional[str]
    description: str
    ttype: str     # debit|credit
    category: str  # income|...|other

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("currency"): d["currency"] = "INR"
        if not d.get("merchant"): d["merchant"] = None
        d["amount"] = float(d.get("amount", 0) or 0)
        return d

# ------------------------------- LLM backends ---------------------------------
class LLMBackend:
    def chat(self, messages: List[Dict[str, str]]) -> str: ...
    def json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]: ...

class OllamaLLM(LLMBackend):
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.ollama, self.err = lazy_import_ollama()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.err: return f"Ollama error: {self.err}"
        try:
            r = self.ollama.chat(model=self.model, messages=messages)
            return r["message"]["content"]
        except Exception as e:
            return f"Error calling Ollama: {e}"

    def json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self.err: return {}
        sys = {"role": "system", "content": (
            "You are a JSON emitter. Output ONLY a single JSON object with keys: "
            "tdate, amount, currency, merchant, description, ttype (debit|credit), "
            "category (income|rent|utilities|groceries|transport|dining|shopping|health|"
            "education|entertainment|travel|fees|emi|investment|savings|other). "
            "Dates in YYYY-MM-DD. No extra text."
        )}
        try:
            r = self.ollama.chat(model=self.model, messages=[sys]+messages)
            txt = (r["message"]["content"] or "").strip()
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1: txt = txt[s:e+1]
            return json.loads(txt)
        except Exception:
            return {}

class GroqLLM(LLMBackend):
    VALID_MODELS = ["llama-3.1-8b-instant"]
    def __init__(self, api_key: str, model: str = ""):
        Groq, err = lazy_import_groq()
        self.err = err
        self.client = None
        if not err:
            try: self.client = Groq(api_key=api_key)
            except Exception as e: self.err = e
        self.model = model.strip() or self.VALID_MODELS[0]

    def _chat_once(self, messages, model):
        r = self.client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return r.choices[0].message.content

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.err: return f"Groq error: {self.err}"
        try:
            return self._chat_once(messages, self.model)
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ["model_decommissioned", "model_not_found", "404", "400"]):
                for alt in self.VALID_MODELS:
                    if alt != self.model:
                        try: return self._chat_once(messages, alt)
                        except Exception: pass
            return f"Error calling Groq: {e}"

    def json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self.err: return {}
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages + [{"role":"system","content":"Reply ONLY with a single JSON object."}],
                temperature=0.1,
                response_format={"type":"json_object"},
            )
            return json.loads(r.choices[0].message.content or "{}")
        except Exception:
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)
                txt = (r.choices[0].message.content or "").strip()
                s, e = txt.find("{"), txt.rfind("}")
                if s != -1 and e != -1: txt = txt[s:e+1]
                return json.loads(txt)
            except Exception:
                return {}

class GeminiLLM(LLMBackend):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai, err = lazy_import_gemini()
        self.err = err
        self.genai = genai
        self.client = None
        self.model_name = model
        if not err:
            try:
                self.genai.configure(api_key=api_key)
                self.client = self.genai.GenerativeModel(
                    self.model_name,
                    generation_config=self.genai.GenerationConfig(temperature=0.2),
                )
            except Exception as e:
                self.err = e

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.err: return f"Gemini error: {self.err}"
        try:
            parts = []
            for m in messages:
                prefix = "User:" if m["role"] == "user" else "Assistant:" if m["role"] == "assistant" else "System:"
                parts.append(f"{prefix} {m['content']}")
            prompt = "\n".join(parts)
            r = self.client.generate_content(prompt)
            return (r.text or "").strip()
        except Exception as e:
            return f"Error calling Gemini: {e}"

    def json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self.err: return {}
        try:
            gcfg = self.genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
            cjson = self.genai.GenerativeModel(self.model_name, generation_config=gcfg)
            parts = []
            for m in messages:
                prefix = "User:" if m["role"] == "user" else "Assistant:" if m["role"] == "assistant" else "System:"
                parts.append(f"{prefix} {m['content']}")
            prompt = "\n".join(parts)
            r = cjson.generate_content(prompt)
            txt = (r.text or "").strip()
            return json.loads(txt)
        except Exception:
            try:
                r = self.client.generate_content("\n".join([f"{m['role']}:{m['content']}" for m in messages]) + "\nReply ONLY with JSON.")
                txt = (r.text or "").strip()
                s, e = txt.find("{"), txt.rfind("}")
                if s != -1 and e != -1: txt = txt[s:e+1]
                return json.loads(txt)
            except Exception:
                return {}

# ------------------------------- LLM utilities --------------------------------
def classify_text_to_txn(llm: LLMBackend, text: str) -> FinanceItem:
    today = dt.date.today().isoformat()
    messages = [
        {"role": "system", "content": "Extract ONE transaction. If date missing, use today's date. Reply ONLY JSON."},
        {"role": "user", "content": f"Today is {today}. Text: {text}"},
    ]
    data = llm.json(messages)
    return FinanceItem(
        tdate=str(data.get("tdate") or today),
        amount=float(data.get("amount") or 0),
        currency=str(data.get("currency") or "INR"),
        merchant=(data.get("merchant") or None),
        description=str(data.get("description") or "Transaction"),
        ttype=str(data.get("ttype") or "debit"),
        category=str(data.get("category") or "other"),
    )

def summarize_budget_with_llm(llm: LLMBackend, monthly_summary: Dict[str, float], goals: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Given my monthly category totals (INR) and goals, suggest 3 prioritized actions.\n"
            f"Totals: {json.dumps(monthly_summary)}\n"
            f"Goals:  {json.dumps(goals)}\n"
            "Keep to 120 words, bullets."}
    ]
    return llm.chat(messages)

def month_key(d: str) -> str:
    try: return dt.datetime.fromisoformat(d).strftime("%Y-%m")
    except Exception: return "unknown"

def monthly_totals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if df.empty: return out
    for _, r in df.iterrows():
        k = month_key(str(r["tdate"]))
        cat = (r["category"] or "other").strip()
        amt = float(r["amount"] or 0)
        if str(r["ttype"]).lower() == "credit":
            amt = 0.0
        out.setdefault(k, {})
        out[k][cat] = out[k].get(cat, 0.0) + amt
    return out

# ---------------------------- Symbol detection --------------------------------
def _normalize_single(word: str) -> str:
    w = word.strip().upper()
    if w.endswith((".NS", ".BO")): return w
    if word.lower() in NAME_TO_TICKER: return NAME_TO_TICKER[word.lower()]
    if re.fullmatch(r"[A-Z]{2,10}", w): return w + ".NS"
    return ""

def detect_symbols(text: str) -> List[str]:
    low = text.lower()
    words = re.findall(r"[a-zA-Z]+", low)
    phrases = set()
    for i in range(len(words)):
        if i+1 < len(words): phrases.add(words[i]+" "+words[i+1])
        if i+2 < len(words): phrases.add(words[i]+" "+words[i+1]+" "+words[i+2])
    seen, out = set(), []
    for ph in phrases:
        if ph in NAME_TO_TICKER:
            sym = NAME_TO_TICKER[ph]
            if sym not in seen: seen.add(sym); out.append(sym)
    tokens = re.findall(r"[A-Za-z\.]{2,20}", text)
    for tok in tokens:
        sym = NAME_TO_TICKER.get(tok.lower()) or _normalize_single(tok)
        if sym and sym not in seen: seen.add(sym); out.append(sym)
    for name, sym in NAME_TO_TICKER.items():
        if name in low and sym not in seen: seen.add(sym); out.append(sym)
    return out[:5]

def expand_candidates(raw: str) -> List[str]:
    s = raw.strip()
    cands = []
    if s.upper().endswith((".NS",".BO")):
        cands.append(s.upper())
    else:
        if s.lower() in NAME_TO_TICKER: cands.append(NAME_TO_TICKER[s.lower()])
        up = re.sub(r"[^A-Za-z]", "", s).upper()
        if 2 <= len(up) <= 10:
            cands += [up + ".NS", up + ".BO"]
        if "NTPC" in up and "GREEN" in up: cands += ["NTPCGREEN.NS", "NTPCGREEN.BO", "NTPC.NS"]
        if "ADANI" in up and "GREEN" in up: cands += ["ADANIGREEN.NS", "ADANIGREEN.BO"]
    out, seen = [], set()
    for x in cands:
        if x and x not in seen: out.append(x); seen.add(x)
    return out

# ---------------------------- Market data (TD/YF) -----------------------------
def fetch_quotes_yahoo(symbols: List[str]) -> List[Dict[str, Any]]:
    yf, err = lazy_import_yfinance()
    if err: raise err
    now = dt.datetime.now(INDIA_TZ)
    rows = []
    for raw in symbols:
        good = None
        for s in (expand_candidates(raw) or [raw]):
            price = None; currency = "INR"; exchange = ""; as_of = None
            try:
                hist = yf.download(s, period="1d", interval="1m", progress=False, threads=False)
                if not hist.empty:
                    last = hist.iloc[-1]
                    price = float(last["Close"])
                    ts = last.name.to_pydatetime()
                    if ts.tzinfo is None: ts = ts.replace(tzinfo=dt.timezone.utc)
                    as_of = ts.astimezone(INDIA_TZ)
                if price is None:
                    t = yf.Ticker(s); fi = getattr(t, "fast_info", {}) or {}
                    price = fi.get("last_price")
                    currency = fi.get("currency") or currency
                    exchange = fi.get("exchange") or exchange
                    if price is not None and as_of is None: as_of = now
                if price is None:
                    t = yf.Ticker(s); h = t.history(period="1d")
                    if not h.empty:
                        price = float(h["Close"].iloc[-1]); as_of = now
            except Exception:
                price = None
            if price is not None:
                if not exchange:
                    try:
                        t = yf.Ticker(s); fi2 = getattr(t, "fast_info", {}) or {}
                        exchange = fi2.get("exchange") or ""; currency = fi2.get("currency") or currency
                    except Exception: pass
                age = (now - as_of).total_seconds()/60.0 if as_of else None
                good = {"input": raw, "symbol": s, "price": float(price), "currency": currency,
                        "exchange": exchange, "as_of": as_of.isoformat() if as_of else None,
                        "age_minutes": round(age, 2) if age is not None else None,
                        "source": "yahoo"}
                break
        if good: rows.append(good)
    return rows

def fetch_quotes_twelvedata(symbols: List[str], api_key: str) -> List[Dict[str, Any]]:
    rows = []
    now = dt.datetime.now(INDIA_TZ)
    session = requests.Session()
    session.headers.update({"User-Agent": "pf-chatbot/1.0"})

    def td_formats(sym: str) -> List[str]:
        base = sym.upper().replace(".NS","").replace(".BO","")
        fmts = [
            f"{base}:NSE", f"{base}.NSE", f"NSE:{base}",
            f"{base}:BSE", f"{base}.BSE", f"BSE:{base}",
        ]
        seen, out = set(), []
        for f in fmts:
            if f not in seen:
                out.append(f); seen.add(f)
        return out

    for raw in symbols:
        fresh_row = None
        for s in (expand_candidates(raw) or [raw]):
            for sym_td in td_formats(s):
                try:
                    # quote endpoint
                    url = "https://api.twelvedata.com/quote"
                    q = {"symbol": sym_td, "apikey": api_key}
                    r = session.get(url, params=q, timeout=8)
                    if r.ok:
                        data = r.json()
                        if isinstance(data, dict) and data.get("price"):
                            price = float(data["price"])
                            ts = data.get("timestamp") or data.get("datetime")
                            if ts:
                                try:
                                    as_of = dt.datetime.fromisoformat(ts)
                                except Exception:
                                    as_of = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                                as_of = as_of.replace(tzinfo=INDIA_TZ)
                            else:
                                as_of = now
                            age_min = (now - as_of).total_seconds()/60.0
                            if age_min <= 1440:
                                fresh_row = {
                                    "input": raw, "symbol": s, "price": price, "currency": "INR",
                                    "exchange": "NSE/BSE", "as_of": as_of.isoformat(),
                                    "age_minutes": round(age_min, 2), "source": "twelvedata"
                                }
                                break
                    if fresh_row is None:
                        # 1-minute last candle
                        url = "https://api.twelvedata.com/time_series"
                        q = {"symbol": sym_td, "interval": "1min", "outputsize": 1, "apikey": api_key}
                        r = session.get(url, params=q, timeout=8)
                        if r.ok:
                            js = r.json()
                            vals = js.get("values") or []
                            if vals:
                                last = vals[0]
                                price = float(last["close"])
                                ts = last.get("datetime")
                                if ts:
                                    try:
                                        as_of = dt.datetime.fromisoformat(ts)
                                    except Exception:
                                        as_of = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                                    as_of = as_of.replace(tzinfo=INDIA_TZ)
                                else:
                                    as_of = now
                                age_min = (now - as_of).total_seconds()/60.0
                                if age_min <= 1440:
                                    fresh_row = {
                                        "input": raw, "symbol": s, "price": price, "currency": "INR",
                                        "exchange": "NSE/BSE", "as_of": as_of.isoformat(),
                                        "age_minutes": round(age_min, 2), "source": "twelvedata"
                                    }
                                    break
                except Exception:
                    pass
            if fresh_row:
                rows.append(fresh_row)
                break
    return rows

def get_quotes(symbols: List[str], provider: str, td_key: str) -> List[Dict[str, Any]]:
    if not symbols:
        return []
    rows: List[Dict[str, Any]] = []
    td_enabled = (provider == "Twelve Data (recommended)" and td_key.strip() and td_key.strip().lower() != "demo")

    if td_enabled:
        rows = fetch_quotes_twelvedata(symbols, td_key)
        if not rows:
            rows = fetch_quotes_yahoo(symbols)
    else:
        if provider == "Twelve Data (recommended)" and (not td_key.strip() or td_key.strip().lower() == "demo"):
            st.warning("Twelve Data selected, but API key is missing or looks like a demo key. Falling back to Yahoo (often delayed).")
        rows = fetch_quotes_yahoo(symbols)

    # drop anything older than 24h just in case
    now = dt.datetime.now(INDIA_TZ)
    fresh = []
    for r in rows:
        try:
            as_of = dt.datetime.fromisoformat(r["as_of"])
            age = (now - as_of).total_seconds() / 60.0
            if age <= 1440:
                fresh.append(r)
        except Exception:
            fresh.append(r)
    return fresh

# ------------------------------- UI / Sidebar ---------------------------------
init_db()
st.title("💸 Personal Finance Chatbot — Llama 3")

# Prefer Streamlit secrets, then environment vars, then blank
def _get_secret(name: str, env_default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, env_default))
    except Exception:
        return os.getenv(name, env_default)

with st.sidebar:
    st.subheader("Provider")
    provider = st.selectbox("Choose backend", ["Ollama (local)", "Gemini (Google AI)", "Groq (cloud)"], index=1)

    if provider == "Ollama (local)":
        model_name = st.text_input("Local model (Ollama)", "llama3")
        llm: LLMBackend = OllamaLLM(model=model_name)
        st.caption("Run `ollama serve` and `ollama pull llama3` once.")

    elif provider == "Gemini (Google AI)":
        gemini_key_default = _get_secret("GEMINI_API_KEY", "AIzaSyA_Sr333AUTwbQMecIxe5M9z0ZpBNri8xo")
        gem_model_default = _get_secret("GEMINI_MODEL", "gemini-1.5-flash")
        gemini_key = st.text_input("GEMINI_API_KEY", gemini_key_default, type="password")
        gem_model = st.text_input("Gemini model", gem_model_default)
        llm = GeminiLLM(api_key=gemini_key, model=gem_model)
        st.caption("Get a key at https://aistudio.google.com/")

    else:  # Groq
        groq_key_default = _get_secret("GROQ_API_KEY", "")
        llama_model_default = _get_secret("LLAMA_MODEL", "llama-3.1-8b-instant")
        groq_key = st.text_input("GROQ_API_KEY", groq_key_default, type="password")
        model_name = st.text_input("Llama model (Groq)", llama_model_default)
        llm = GroqLLM(api_key=groq_key, model=model_name)
        st.caption("Use a supported model like `llama-3.1-8b-instant`.")

    st.divider()
    st.subheader("Goals (optional)")
    income = st.number_input("Monthly income (INR)", min_value=0, value=0, step=1000)
    save_goal = st.number_input("Savings goal (INR/month)", min_value=0, value=0, step=1000)
    debt_goal = st.text_input("Debt payoff focus", value="")

    st.divider()
    st.subheader("Market data")
    md_provider = st.selectbox("Data source", ["Twelve Data (recommended)", "Yahoo/yfinance (free, delayed)"])
    td_key_default = _get_secret("TWELVE_DATA_API_KEY", "5b0be88afbb840da9acc8d0e280f6e63")
    td_key = st.text_input("TWELVE_DATA_API_KEY (near real-time)", td_key_default, type="password")
    use_live = st.checkbox("Enable live stock prices", value=True)
    force_tickers = st.text_input("Force tickers (comma-separated, e.g., ADANIGREEN.NS,NTPC.NS)", value="")
    custom_map = st.text_area("Custom name→ticker map (one per line: name = SYMBOL.SFX)",
                              value="adani green energy = ADANIGREEN.NS\nntpc green = NTPCGREEN.NS\nbharati airtel = BHARTIARTL.NS\nbharati = BHARTIARTL.NS\nbharti = BHARTIARTL.NS")
    for line in custom_map.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip().lower(), v.strip().upper()
            if k and v: NAME_TO_TICKER[k] = v

tabs = st.tabs(["💬 Chat", "🧾 Transactions", "📊 Reports"])

# ---------------------------------- Chat Tab ----------------------------------
with tabs[0]:
    st.markdown("> Educational guidance only; not financial/tax advice.")

    if "chat" not in st.session_state:
        st.session_state.chat = [{"role":"system","content":SYSTEM_PROMPT}]

    for m in st.session_state.chat[1:]:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask about budgets, EMIs/SIPs, or stock tickers (e.g., ADANIGREEN.NS, RELIANCE)…")
    if prompt:
        st.session_state.chat.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        live_context = ""
        if use_live:
            try:
                symbols = [s.strip() for s in force_tickers.split(",") if s.strip()] if force_tickers.strip() else detect_symbols(prompt)
                st.caption(f"🔎 Detected symbols: {symbols}" if symbols else "🔎 No symbols detected")
                quotes = get_quotes(symbols, md_provider, td_key) if symbols else []
                if quotes:
                    st.write(f"**Market data** — source: {quotes[0].get('source','unknown')}")
                    qdf = pd.DataFrame(quotes); st.dataframe(qdf)
                    limit = 3 if quotes[0].get("source") == "twelvedata" else 15
                    olds = [q for q in quotes if q.get("age_minutes") and q["age_minutes"] > limit]
                    if olds:
                        st.warning(f"Some quotes are older than {limit} minutes. Check symbol format and your TD key.")
                    live_lines = [f"{q['symbol']}: {q['price']} {q.get('currency','INR')} (as of {q['as_of']})" for q in quotes]
                    live_context = "\n\n[Live market context]\n" + "\n".join(live_lines)
                else:
                    st.warning("No fresh quotes returned. Try forcing a ticker like `BHARTIARTL.NS` or verify the Twelve Data key.")
            except Exception as e:
                st.warning(f"Live quotes unavailable: {e}.")

        with st.chat_message("assistant"):
            msgs = st.session_state.chat[-10:].copy()
            if live_context:
                msgs[-1] = {"role":"user","content": msgs[-1]["content"] + live_context +
                            "\n\nUse the above live quotes for any calculations or comparisons."}
            reply = llm.chat(msgs)
            st.write(reply)
        st.session_state.chat.append({"role":"assistant","content":reply})

# ------------------------------ Transactions Tab ------------------------------
with tabs[1]:
    st.subheader("Add or import transactions")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.write("**Quick capture (NL → structured)**")
        text_txn = st.text_input("e.g., Paid ₹250 on Zomato for lunch yesterday")
        if st.button("Parse & Save"):
            if text_txn.strip():
                try:
                    item = classify_text_to_txn(llm, text_txn)
                    insert_txn(item.to_row())
                    st.success(f"Saved: {item.ttype} ₹{item.amount} → {item.category} on {item.tdate}")
                except Exception as e:
                    st.error(f"Could not parse: {e}")

    with c2:
        st.write("**CSV import**")
        st.caption("Columns (any case): date, description, amount, (optional) type, category, merchant, currency")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
                cols = {c.lower().strip(): c for c in df.columns}
                def pick(*names):
                    for n in names:
                        if n in cols: return cols[n]
                    return None
                c_date = pick("date","tdate"); c_desc = pick("description","narration")
                c_amt  = pick("amount","amt");  c_type = pick("type","ttype")
                c_cat  = pick("category");      c_mer  = pick("merchant","payee")
                c_cur  = pick("currency")
                st.dataframe(df.head(50))

                if st.button("Categorize (LLM) & Save top 200"):
                    ok, fail = 0, 0
                    for row in df.head(200).to_dict(orient="records"):
                        parts=[]
                        if c_date and row.get(c_date): parts.append(str(row[c_date]))
                        if c_desc and row.get(c_desc): parts.append(str(row[c_desc]))
                        if c_mer  and row.get(c_mer):  parts.append(f"merchant {row[c_mer]}")
                        if c_amt  and row.get(c_amt) is not None: parts.append(f"amount {row[c_amt]}")
                        if c_type and row.get(c_type): parts.append(f"type {row[c_type]}")
                        if c_cat  and row.get(c_cat):  parts.append(f"category {row[c_cat]}")
                        if c_cur  and row.get(c_cur):  parts.append(f"currency {row[c_cur]}")
                        try:
                            item = classify_text_to_txn(llm, " | ".join(map(str, parts)))
                            payload = item.to_row(); payload["source"] = "csv"
                            insert_txn(payload); ok += 1
                        except Exception:
                            fail += 1
                    st.success(f"Imported {ok} rows. Failed: {fail}")
            except Exception as e:
                st.error(f"CSV error: {e}")

    st.divider(); st.subheader("Current transactions")
    try:
        st.dataframe(read_txns())
    except Exception as e:
        st.error(f"Read error: {e}")

# --------------------------------- Reports Tab --------------------------------
with tabs[2]:
    st.subheader("Monthly overview")
    try:
        tx = read_txns()
        if tx.empty:
            st.info("No data yet. Add a few transactions first.")
        else:
            agg = monthly_totals(tx)
            months = sorted(agg.keys())
            if months:
                m = st.selectbox("Pick month", months, index=0)
                cats = agg.get(m, {})
                if cats:
                    plt, err = lazy_import_matplotlib()
                    if err:
                        st.warning(f"Charts disabled: {err}. Try `pip install matplotlib`.")
                    else:
                        fig = plt.figure()
                        plt.title(f"Spend by category — {m}")
                        plt.bar(list(cats.keys()), list(cats.values()))
                        plt.xticks(rotation=45, ha="right"); plt.ylabel("INR")
                        st.pyplot(fig)

                        goals = {"income": income, "target_saving": save_goal, "debt_focus": debt_goal}
                        if st.button("Get Llama advice for this month"):
                            st.markdown(summarize_budget_with_llm(llm, cats, goals))
    except Exception as e:
        st.error(f"Reports error: {e}")
