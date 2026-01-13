# 🤖 Agentic Backtesting System

> **KB-Aware · Deterministic · Human-in-the-Loop · Auditable**

A production-grade, multi-agent backtesting framework that bridges natural language and quantitative finance through a deterministic, auditable pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Core Philosophy

```
The system may pause, wait, or switch to planning mode —
but it will NEVER execute unless it is fully certain,
validated, and supported by deterministic tools.
```

This is not a traditional backtesting tool. It's an **agentic orchestration system** where multiple specialized agents collaborate to translate English intent into reproducible, verifiable backtests.

---

## 🏗️ Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (English)                        │
│                 "Run SuperTrend on NIFTY from Jan 1-31"             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: PARSER AGENT                                               │
│  ├── Rule-based NLP (no ML inference)                               │
│  ├── Extracts: Strategy, Dataset, Date Range                        │
│  └── Outputs: Confidence Score (0-100%)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ STEP 2: AMBIGUITY │           │ STEP 3: HUMAN     │
        │ GATE (≥50%)       │──FAIL────▶│ CLARIFICATION     │
        │ ✓ PASSED          │           │ (English only)    │
        └───────────────────┘           └───────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: STRATEGY BUILDER AGENT                                     │
│  ├── Maps intent → KB Artifact IDs                                  │
│  ├── Resolves: strat_supertrend_001, dataset_nifty_fut_001          │
│  └── Links required indicators                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: VALIDATION AGENT                                           │
│  ├── Checks data availability                                       │
│  ├── Verifies indicator compatibility                               │
│  └── Ensures all required components exist in KB                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: CAPABILITY CHECK                                           │
│  ├── Can we actually execute this?                                  │
│  └── If NO → Enter PLANNING MODE (Step 8B)                          │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ╔═══════════════════════════════════════╗
        ║  STEP 8A: STRATEGY FREEZE 🔒          ║
        ║  ─────────────────────────────────    ║
        ║  Configuration becomes IMMUTABLE      ║
        ║  Hash: a1b2c3d4e5f6...                ║
        ║  No agent can modify post-approval    ║
        ╚═══════════════════════════════════════╝
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 9-10: EXECUTION ENGINE                                        │
│  ├── Loads frozen config                                            │
│  ├── Computes indicators deterministically                          │
│  ├── Simulates trades (Signal@T → Execute@T+1)                      │
│  └── Generates: Trades, Equity Curve, Metrics, Audit Hash           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔒 Determinism & Auditability
- **Execution Hash** - Every backtest produces a cryptographic hash for reproducibility
- **Freeze Boundary** - Configuration becomes immutable before execution
- **No ML in Logic** - All execution is rule-based, ensuring consistent results

### 🧠 Knowledge Base Architecture
- **Strategy Artifacts** - JSON definitions with entry/exit rules, parameters
- **Indicator Library** - Versioned indicator implementations (SuperTrend, RSI, MACD, Bollinger)
- **Dataset Registry** - Metadata linking to OHLCV data files

### 📊 Professional Metrics
| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted returns (annualized) |
| Sortino Ratio | Downside deviation only |
| Max Drawdown | Peak-to-trough decline |
| CAGR | Compound annual growth rate |
| Profit Factor | Gross profit / Gross loss |
| Expectancy | Average expected profit per trade |

### 🛡️ Fallback Mechanisms
- **Ambiguity Gate** - Pauses when confidence < 50%
- **Planning Mode** - If execution impossible, suggests next steps
- **Human Clarification** - Asks in English, never for parameters

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Aruni20/Agentic-Backtesting-System.git
cd Agentic-Backtesting-System

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Open **http://localhost:8501** and try:
```
Run SuperTrend on NIFTY from January 1st to January 31st 2026
```

---

## 📁 Project Structure

```
├── app.py                 # Streamlit UI with multi-agent orchestration
├── src/
│   ├── agents.py          # Parser, Builder, Validation agents
│   ├── engine.py          # Deterministic execution engine
│   └── kb_interface.py    # Knowledge Base read-only interface
├── kb/
│   ├── strategies/        # Strategy definitions (JSON)
│   ├── indicators/        # Indicator definitions (JSON)
│   └── datasets/          # Dataset metadata (JSON)
├── data/
│   └── nifty_futures_sample.csv  # Sample OHLCV data
└── requirements.txt
```

---

## 🎨 Available Strategies

| Strategy | Type | Indicators |
|----------|------|------------|
| SuperTrend | Trend Following | ATR-based trailing stop |
| RSI Reversal | Momentum | RSI (14) |
| MACD Crossover | Trend | MACD (12,26,9) |
| Bollinger Bands | Volatility | 20-period SMA ± 2σ |
| MA Crossover | Trend | SMA (Fast/Slow) |

---

## 🔧 Production Features

- ✅ **State Machine UI** - Not a chatbot, an airlock between English and code
- ✅ **Progress Visualization** - Step-by-step pipeline execution with timing
- ✅ **Export Capabilities** - Download trades (CSV), metrics (JSON), full reports
- ✅ **Interactive Charts** - Plotly candlesticks with trade markers
- ✅ **Date Range Picker** - Configurable backtest periods
- ✅ **Sidebar KB Browser** - View available strategies and datasets

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with ❤️ for deterministic, auditable quantitative finance</b>
</p>
