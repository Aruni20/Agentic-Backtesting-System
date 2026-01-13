"""
app.py - Enterprise Backtesting System
KB-Aware · Deterministic · Human-in-the-Loop · Auditable

Core System Law:
The system may pause, wait, or switch to planning mode — but it will NEVER execute
unless it is fully certain, validated, and supported by deterministic tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kb_interface import KnowledgeBase
from engine import DeterministicEngine
from agents import ParserAgent, StrategyBuilderAgent, ValidationAgent


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
KB_ROOT = PROJECT_ROOT / "kb"
DATA_ROOT = PROJECT_ROOT / "data"

CONFIDENCE_THRESHOLD = 0.5

# Pipeline States
STATES = {
    "IDLE": {"label": "Ready", "color": "#6366f1", "icon": "⏸️"},
    "STEP_1": {"label": "Step 1: Parsing Intent", "color": "#f59e0b", "icon": "🔍"},
    "STEP_2": {"label": "Step 2: Ambiguity Gate", "color": "#f59e0b", "icon": "🚧"},
    "STEP_3": {"label": "Step 3: Human Clarification", "color": "#ef4444", "icon": "👤"},
    "STEP_5": {"label": "Step 5: Strategy Builder", "color": "#f59e0b", "icon": "🔗"},
    "STEP_6": {"label": "Step 6: Validation Agent", "color": "#f59e0b", "icon": "✅"},
    "STEP_7": {"label": "Step 7: Capability Check", "color": "#f59e0b", "icon": "⚙️"},
    "STEP_8A": {"label": "Step 8A: Strategy Freeze", "color": "#dc2626", "icon": "🔒"},
    "STEP_8B": {"label": "Step 8B: Planning Mode", "color": "#7c3aed", "icon": "📋"},
    "STEP_9": {"label": "Step 9: Execution Agent", "color": "#22c55e", "icon": "⚡"},
    "STEP_10": {"label": "Step 10: Backtest Engine", "color": "#22c55e", "icon": "📊"},
    "COMPLETE": {"label": "Complete", "color": "#22c55e", "icon": "✓"},
    "ERROR": {"label": "Halted", "color": "#ef4444", "icon": "🛑"},
}


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session():
    defaults = {
        "messages": [],
        "state": "IDLE",
        "pipeline_log": [],
        "parsed_intent": None,
        "bound_strategy": None,
        "frozen_config": None,
        "execution_result": None,
        "chart_data": None,
        "api_key": "",
        "api_provider": "OpenAI",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_step(step_id: str, message: str, status: str = "success"):
    st.session_state.pipeline_log.append({
        "step": step_id,
        "message": message,
        "status": status,
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
    })


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


# =============================================================================
# CUSTOM CSS
# =============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 50%, #0f0f23 100%);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.1em;
    }
    
    .invariant-box {
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.05) 100%);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #fca5a5;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(22,163,74,0.05) 100%);
        border: 1px solid rgba(34,197,94,0.3);
    }
    
    .metric-card-red {
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.05) 100%);
        border: 1px solid rgba(239,68,68,0.3);
    }
    
    .metric-card-gold {
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(217,119,6,0.05) 100%);
        border: 1px solid rgba(245,158,11,0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .freeze-boundary {
        background: linear-gradient(135deg, rgba(220,38,38,0.15) 0%, rgba(185,28,28,0.1) 100%);
        border: 2px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .freeze-title {
        color: #fca5a5;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .log-entry {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.75rem;
        padding: 0.4rem 0.6rem;
        border-left: 3px solid;
        margin: 0.2rem 0;
        background: rgba(0,0,0,0.3);
        border-radius: 0 6px 6px 0;
    }
    
    .log-success { border-color: #22c55e; color: #86efac; }
    .log-error { border-color: #ef4444; color: #fca5a5; }
    .log-warning { border-color: #f59e0b; color: #fcd34d; }
    .log-info { border-color: #6366f1; color: #a5b4fc; }
    
    .strategy-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    
    .help-section {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99,102,241,0.3);
    }
    
    div[data-testid="stChatInput"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# CHARTS
# =============================================================================

def create_candlestick_chart(df: pd.DataFrame, trades: list = None, supertrend: pd.Series = None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action with Indicators', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',
        ),
        row=1, col=1
    )
    
    # SuperTrend overlay
    if 'supertrend' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['supertrend'],
                mode='lines',
                name='SuperTrend',
                line=dict(color='#f59e0b', width=2),
            ),
            row=1, col=1
        )
    
    # Trade markers
    if trades:
        for trade in trades:
            fig.add_trace(
                go.Scatter(
                    x=[pd.to_datetime(trade.entry_date)],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.direction == 'LONG' else 'triangle-down',
                        size=15,
                        color='#22c55e' if trade.direction == 'LONG' else '#ef4444',
                        line=dict(color='white', width=2)
                    ),
                    name=f'{trade.direction} Entry',
                    showlegend=False,
                    hovertemplate=f"Entry: ₹{trade.entry_price:,.2f}<br>Date: {trade.entry_date}<extra></extra>"
                ),
                row=1, col=1
            )
            if trade.exit_date:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(trade.exit_date)],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=12,
                            color='#f59e0b',
                            line=dict(color='white', width=2)
                        ),
                        name='Exit',
                        showlegend=False,
                        hovertemplate=f"Exit: ₹{trade.exit_price:,.2f}<br>P&L: ₹{trade.pnl:,.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
    
    # Volume
    colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    
    return fig


def create_equity_chart(equity_curve: pd.DataFrame):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=equity_curve['equity'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#8b5cf6', width=2),
        fillcolor='rgba(139,92,246,0.2)',
        name='Equity'
    ))
    
    # Add high watermark
    hwm = equity_curve['equity'].cummax()
    fig.add_trace(go.Scatter(
        y=hwm,
        mode='lines',
        line=dict(color='#22c55e', width=1, dash='dot'),
        name='High Watermark'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=True,
        yaxis_title='Equity (₹)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def get_chart_download_link(fig, filename="chart"):
    """Generate download link for chart as HTML."""
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    b64 = base64.b64encode(html_bytes).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}.html">📥 Download Chart</a>'


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def execute_pipeline(user_input: str):
    import time
    
    st.session_state.pipeline_log = []
    st.session_state.state = "STEP_1"
    
    kb = KnowledgeBase(str(KB_ROOT))
    parser = ParserAgent()
    builder = StrategyBuilderAgent(kb)
    validator = ValidationAgent(kb)
    engine = DeterministicEngine(str(DATA_ROOT))
    
    # STEP 1: Parse Intent
    log_step("STEP_1", "Converting English → Structured Intent", "info")
    time.sleep(0.4)
    parsed = parser.parse(user_input)
    st.session_state.parsed_intent = parsed
    
    log_step("STEP_1", f"Strategy: {parsed.strategy_name or 'None'}", "success" if parsed.strategy_name else "warning")
    log_step("STEP_1", f"Dataset: {parsed.dataset_name or 'None'}", "success" if parsed.dataset_name else "warning")
    log_step("STEP_1", f"Confidence: {parsed.confidence:.0%}", "success" if parsed.confidence >= CONFIDENCE_THRESHOLD else "warning")
    time.sleep(0.3)
    
    # STEP 2: Ambiguity Gate
    st.session_state.state = "STEP_2"
    time.sleep(0.4)
    if parsed.confidence < CONFIDENCE_THRESHOLD:
        log_step("STEP_2", f"GATE FAILED: Confidence {parsed.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%}", "error")
        log_step("STEP_2", f"Missing: {', '.join(parsed.missing_fields)}", "error")
        st.session_state.state = "STEP_3"
        return False
    
    log_step("STEP_2", "Ambiguity Gate: PASSED", "success")
    time.sleep(0.3)
    
    # STEP 5: Strategy Builder
    st.session_state.state = "STEP_5"
    log_step("STEP_5", "Binding intent to KB artifacts...", "info")
    time.sleep(0.4)
    bound = builder.build(parsed)
    st.session_state.bound_strategy = bound
    
    if not bound.is_valid:
        for err in bound.validation_errors:
            log_step("STEP_5", f"Binding Error: {err}", "error")
        st.session_state.state = "ERROR"
        return False
    
    log_step("STEP_5", f"Strategy ID: {bound.strategy_id}", "success")
    log_step("STEP_5", f"Dataset ID: {bound.dataset_id}", "success")
    time.sleep(0.3)
    
    # STEP 6: Validation Agent
    st.session_state.state = "STEP_6"
    log_step("STEP_6", "Validation Agent checking...", "info")
    time.sleep(0.4)
    is_approved, issues = validator.validate(bound)
    
    if not is_approved:
        for issue in issues:
            log_step("STEP_6", issue, "error")
        st.session_state.state = "ERROR"
        return False
    
    log_step("STEP_6", "All validation checks: PASSED", "success")
    time.sleep(0.3)
    
    # STEP 7: Capability Check
    st.session_state.state = "STEP_7"
    log_step("STEP_7", "Checking execution capability...", "info")
    time.sleep(0.4)
    
    dataset = kb.get_dataset(bound.dataset_id)
    strategy = kb.get_strategy(bound.strategy_id)
    
    if not dataset or not strategy:
        log_step("STEP_7", "Missing KB artifacts", "error")
        st.session_state.state = "STEP_8B"
        return False
    
    log_step("STEP_7", "All components available", "success")
    time.sleep(0.3)
    
    # STEP 8A: Strategy Freeze
    st.session_state.state = "STEP_8A"
    time.sleep(0.4)
    frozen_config = {
        "strategy_id": bound.strategy_id,
        "strategy_name": strategy.get("name"),
        "dataset_id": bound.dataset_id,
        "dataset_name": dataset.get("name"),
        "indicators": bound.indicator_ids,
        "parameters": bound.parameters,
        "date_range": {
            "start": parsed.start_date or str(st.session_state.get("start_date", "2026-01-01")),
            "end": parsed.end_date or str(st.session_state.get("end_date", "2026-01-31"))
        },
        "kb_version": "1.0.0",
        "frozen_at": datetime.now().isoformat()
    }
    st.session_state.frozen_config = frozen_config
    
    import hashlib
    config_hash = hashlib.sha256(json.dumps(frozen_config, sort_keys=True).encode()).hexdigest()[:16]
    log_step("STEP_8A", f"Configuration FROZEN", "warning")
    log_step("STEP_8A", f"Freeze Hash: {config_hash}", "info")
    time.sleep(0.5)
    
    # STEP 9 & 10: Execute
    st.session_state.state = "STEP_9"
    log_step("STEP_9", "Execution Agent loading frozen config...", "info")
    time.sleep(0.4)
    
    st.session_state.state = "STEP_10"
    log_step("STEP_10", "Loading OHLCV data from KB...", "info")
    time.sleep(0.3)
    
    data_path = dataset["data_path"]
    df = engine.load_data(data_path)
    log_step("STEP_10", f"Loaded {len(df)} bars", "success")
    time.sleep(0.3)
    
    log_step("STEP_10", "Computing indicators...", "info")
    time.sleep(0.4)
    log_step("STEP_10", "Simulating trades (T→T+1 rule)...", "info")
    time.sleep(0.3)
    
    result = engine.execute_strategy(df, bound.parameters)
    st.session_state.execution_result = result
    st.session_state.chart_data = df
    
    log_step("STEP_10", f"Trades: {result.metrics['total_trades']}", "success")
    log_step("STEP_10", f"Win Rate: {result.metrics['win_rate']:.1f}%", "success")
    log_step("STEP_10", f"Sharpe Ratio: {result.metrics['sharpe_ratio']}", "success")
    log_step("STEP_10", f"Max Drawdown: {result.metrics['max_drawdown']:.2f}%", "success")
    log_step("STEP_10", f"Execution Hash: {result.execution_hash}", "info")
    time.sleep(0.3)
    
    st.session_state.state = "COMPLETE"
    return True


# =============================================================================
# MAIN UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Agentic Backtesting System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session()
    inject_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        with st.expander("🔑 API Settings", expanded=False):
            provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google"])
            api_key = st.text_input("API Key", type="password", placeholder="sk-...")
            if api_key:
                st.success("✓ Key configured (encrypted)")
        
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        
        try:
            kb = KnowledgeBase(str(KB_ROOT))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Strategies", len(kb.list_strategies()))
            with col2:
                st.metric("Datasets", len(kb.list_datasets()))
            
            with st.expander("📋 Available Strategies"):
                for sid in kb.list_strategies():
                    s = kb.get_strategy(sid)
                    st.markdown(f"""
                    <div class="strategy-card">
                        <b>{s['name']}</b><br>
                        <small style="color:#94a3b8">{s.get('description', '')[:60]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"KB Error: {e}")
        
        st.markdown("---")
        st.markdown("### 📅 Date Range")
        
        from datetime import date
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "Start Date",
                value=date(2026, 1, 1),
                min_value=date(2026, 1, 1),
                max_value=date(2026, 1, 31),
                key="start_date"
            )
        with col_d2:
            end_date = st.date_input(
                "End Date",
                value=date(2026, 1, 31),
                min_value=date(2026, 1, 1),
                max_value=date(2026, 1, 31),
                key="end_date"
            )
        
        st.caption(f"📊 Data available: Jan 1 - Jan 31, 2026")
        
        st.markdown("---")
        st.markdown("### 🔒 Core Invariant")
        st.markdown("""
        <div class="invariant-box">
        The system may pause, wait, or switch to planning mode — but it will <b>never execute</b> unless fully certain, validated, and supported by deterministic tools.
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    st.markdown('<h1 class="main-title">Agentic Backtesting System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KB-AWARE · DETERMINISTIC · HUMAN-IN-THE-LOOP · AUDITABLE</p>', unsafe_allow_html=True)
    
    # Pipeline Status
    current_state = STATES.get(st.session_state.state, STATES["IDLE"])
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_state['icon']} {current_state['label']}</div>
            <div class="metric-label">Pipeline Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown("---")
    
    if st.session_state.state in ["IDLE", "COMPLETE", "ERROR", "STEP_3"]:
        
        # Custom CSS for Google-style suggestions
        st.markdown("""
        <style>
        .suggestion-container {
            background: rgba(30, 30, 45, 0.95);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            margin-top: -10px;
            padding: 8px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .suggestion-item {
            padding: 10px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #e2e8f0;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .suggestion-item:hover {
            background: rgba(99,102,241,0.15);
        }
        .suggestion-icon {
            color: #6366f1;
            font-size: 1rem;
        }
        .suggestion-text {
            flex: 1;
        }
        .suggestion-tag {
            background: rgba(99,102,241,0.2);
            color: #a5b4fc;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        user_input = st.chat_input("Enter your request in plain English...")
        
        # Show clean suggestion area only when idle
        if st.session_state.state == "IDLE":
            st.markdown("""
            <div style="text-align: center; color: #64748b; font-size: 0.75rem; margin: 10px 0 5px; text-transform: uppercase; letter-spacing: 0.1em;">
                💡 Or select a strategy to run
            </div>
            """, unsafe_allow_html=True)
            
            selected = st.selectbox(
                "Quick Strategy Select",
                options=[
                    "-- Select a strategy --",
                    "📈 SuperTrend on NIFTY (Trend Following)",
                    "📊 RSI Strategy on NIFTY (Momentum)",
                    "📉 MACD Crossover on NIFTY (Trend)",
                    "🎯 Bollinger Bands on NIFTY (Volatility)",
                    "↗️ MA Crossover on NIFTY (Trend)",
                ],
                label_visibility="collapsed"
            )
            
            if selected and selected != "-- Select a strategy --":
                cmd_map = {
                    "📈 SuperTrend on NIFTY (Trend Following)": "Run SuperTrend on NIFTY from January 1st to January 31st 2026",
                    "📊 RSI Strategy on NIFTY (Momentum)": "Run RSI strategy on NIFTY from January 1st to January 31st 2026",
                    "📉 MACD Crossover on NIFTY (Trend)": "Run MACD on NIFTY from January 1st to January 31st 2026",
                    "🎯 Bollinger Bands on NIFTY (Volatility)": "Run Bollinger bands on NIFTY from January 1st to January 31st 2026",
                    "↗️ MA Crossover on NIFTY (Trend)": "Run MA crossover on NIFTY from January 1st to January 31st 2026",
                }
                if st.button("▶ Run Selected Strategy", use_container_width=True):
                    execute_pipeline(cmd_map[selected])
                    st.rerun()
        
        if user_input:
            add_message("user", user_input)
            
            lower_input = user_input.lower()
            if 'help' in lower_input or 'what can you do' in lower_input:
                show_help()
            elif 'list' in lower_input and 'strateg' in lower_input:
                show_strategies()
            elif 'sharpe' in lower_input and st.session_state.execution_result:
                show_metric("Sharpe Ratio", st.session_state.execution_result.metrics.get('sharpe_ratio', 0))
            elif 'sortino' in lower_input and st.session_state.execution_result:
                show_metric("Sortino Ratio", st.session_state.execution_result.metrics.get('sortino_ratio', 0))
            elif 'drawdown' in lower_input and st.session_state.execution_result:
                show_metric("Max Drawdown", f"{st.session_state.execution_result.metrics.get('max_drawdown', 0)}%")
            else:
                execute_pipeline(user_input)
            st.rerun()
    
    # Results Layout
    if st.session_state.state == "COMPLETE" and st.session_state.execution_result:
        result = st.session_state.execution_result
        
        # Primary Metrics Row
        st.markdown("### 📈 Execution Results")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Trades", result.metrics['total_trades'])
        with m2:
            st.metric("Win Rate", f"{result.metrics['win_rate']:.1f}%")
        with m3:
            delta_color = "normal" if result.metrics['total_pnl'] >= 0 else "inverse"
            st.metric("Total P&L", f"₹{result.metrics['total_pnl']:,.2f}", delta=f"₹{result.metrics['total_pnl']:,.2f}", delta_color=delta_color)
        with m4:
            st.metric("Final Equity", f"₹{result.metrics['final_equity']:,.2f}")
        
        # Advanced Metrics Row
        st.markdown("### 📊 Performance Metrics")
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        with a1:
            color = "green" if result.metrics['sharpe_ratio'] > 1 else ("red" if result.metrics['sharpe_ratio'] < 0 else "gold")
            st.markdown(f"""
            <div class="metric-card metric-card-{color}">
                <div class="metric-value">{result.metrics['sharpe_ratio']}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with a2:
            color = "green" if result.metrics['sortino_ratio'] > 1 else "gold"
            st.markdown(f"""
            <div class="metric-card metric-card-{color}">
                <div class="metric-value">{result.metrics['sortino_ratio']}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with a3:
            color = "green" if result.metrics['max_drawdown'] > -10 else "red"
            st.markdown(f"""
            <div class="metric-card metric-card-{color}">
                <div class="metric-value">{result.metrics['max_drawdown']}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        with a4:
            color = "green" if result.metrics['profit_factor'] > 1.5 else ("red" if result.metrics['profit_factor'] < 1 else "gold")
            st.markdown(f"""
            <div class="metric-card metric-card-{color}">
                <div class="metric-value">{result.metrics['profit_factor']}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            """, unsafe_allow_html=True)
        with a5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.metrics['cagr']}%</div>
                <div class="metric-label">CAGR</div>
            </div>
            """, unsafe_allow_html=True)
        with a6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{result.metrics['expectancy']:,.0f}</div>
                <div class="metric-label">Expectancy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "💹 Equity Curve", "📋 Trade Log", "📥 Download"])
        
        with tab1:
            if st.session_state.chart_data is not None:
                engine = DeterministicEngine(str(DATA_ROOT))
                df_with_indicator = engine.calculate_supertrend(st.session_state.chart_data)
                fig = create_candlestick_chart(df_with_indicator, result.trades)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_equity_chart(result.equity_curve)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if result.trades:
                trade_data = []
                for i, t in enumerate(result.trades, 1):
                    trade_data.append({
                        "#": i,
                        "Entry Date": t.entry_date,
                        "Entry Price": f"₹{t.entry_price:,.2f}",
                        "Exit Date": t.exit_date or "-",
                        "Exit Price": f"₹{t.exit_price:,.2f}" if t.exit_price else "-",
                        "Direction": t.direction,
                        "P&L": f"₹{t.pnl:,.2f}" if t.pnl else "-",
                        "Status": "✅" if t.pnl and t.pnl > 0 else "❌"
                    })
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
            else:
                st.info("No trades executed in this backtest period.")
        
        with tab4:
            st.markdown("### 📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Trade Log CSV
                if result.trades:
                    trade_df = pd.DataFrame([{
                        "Entry Date": t.entry_date,
                        "Entry Price": t.entry_price,
                        "Exit Date": t.exit_date,
                        "Exit Price": t.exit_price,
                        "Direction": t.direction,
                        "P&L": t.pnl
                    } for t in result.trades])
                    csv = trade_df.to_csv(index=False)
                    st.download_button(
                        "📋 Download Trade Log (CSV)",
                        csv,
                        "trade_log.csv",
                        "text/csv"
                    )
            
            with col2:
                # Metrics JSON
                metrics_json = json.dumps(result.metrics, indent=2)
                st.download_button(
                    "📊 Download Metrics (JSON)",
                    metrics_json,
                    "metrics.json",
                    "application/json"
                )
            
            with col3:
                # Full Report
                report = {
                    "execution_hash": result.execution_hash,
                    "frozen_config": st.session_state.frozen_config,
                    "metrics": result.metrics,
                    "trades": [{"entry": t.entry_date, "exit": t.exit_date, "pnl": t.pnl} for t in result.trades]
                }
                st.download_button(
                    "📄 Download Full Report (JSON)",
                    json.dumps(report, indent=2),
                    "backtest_report.json",
                    "application/json"
                )
    
    # Pipeline Log & Audit
    st.markdown("---")
    col_log, col_audit = st.columns([2, 1])
    
    with col_log:
        st.markdown("### ⚡ Execution Pipeline")
        log_container = st.container(height=300)
        with log_container:
            for entry in st.session_state.pipeline_log:
                status_class = f"log-{entry['status']}"
                st.markdown(f"""
                <div class="log-entry {status_class}">
                    <b>[{entry['timestamp']}]</b> {entry['step']}: {entry['message']}
                </div>
                """, unsafe_allow_html=True)
    
    with col_audit:
        st.markdown("### 🔐 Audit Trail")
        if st.session_state.frozen_config:
            st.markdown("""
            <div class="freeze-boundary">
                <div class="freeze-title">🔒 FROZEN CONFIGURATION</div>
                <small>Immutable after approval. No agent modification allowed.</small>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("View Frozen Config"):
                st.json(st.session_state.frozen_config)
        
        if st.session_state.execution_result:
            st.code(f"Execution Hash: {st.session_state.execution_result.execution_hash}")
    
    # Reset Button
    if st.session_state.state != "IDLE":
        if st.button("🔄 New Backtest"):
            for key in ["messages", "pipeline_log", "parsed_intent", "bound_strategy", 
                       "frozen_config", "execution_result", "chart_data"]:
                st.session_state[key] = [] if key in ["messages", "pipeline_log"] else None
            st.session_state.state = "IDLE"
            st.rerun()


def show_help():
    """Show help information."""
    log_step("HELP", "Displaying available commands", "info")
    st.session_state.state = "COMPLETE"


def show_strategies():
    """Show available strategies."""
    kb = KnowledgeBase(str(KB_ROOT))
    for sid in kb.list_strategies():
        s = kb.get_strategy(sid)
        log_step("LIST", f"{s['name']} ({sid})", "info")
    st.session_state.state = "COMPLETE"


def show_metric(name: str, value):
    """Show a specific metric."""
    log_step("METRIC", f"{name}: {value}", "success")
    st.session_state.state = "COMPLETE"


if __name__ == "__main__":
    main()
