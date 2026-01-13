"""
engine.py - Deterministic Backtesting Engine

Core Invariant:
    Signals are evaluated at time T.
    Trades are executed at T+1 open.
    No randomness. No learning. No look-ahead.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Immutable trade record."""
    entry_date: str
    entry_price: float
    exit_date: Optional[str]
    exit_price: Optional[float]
    direction: str  # 'LONG' or 'SHORT'
    pnl: Optional[float] = None


@dataclass
class ExecutionResult:
    """Result of a deterministic backtest execution."""
    success: bool
    error_message: Optional[str]
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]
    execution_hash: str  # For audit trail


class DeterministicEngine:
    """
    Deterministic execution engine.
    No AI. No randomness. Pure rule-based execution.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load OHLCV data from CSV."""
        full_path = self.data_root.parent / data_path
        df = pd.read_csv(full_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def calculate_supertrend(
        self,
        df: pd.DataFrame,
        atr_period: int = 10,
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        Returns DataFrame with 'supertrend' and 'direction' columns.
        direction: 1 = UP (bullish), -1 = DOWN (bearish)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = tr.rolling(window=atr_period).mean()

        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # First valid index after ATR calculation
        start_idx = atr_period

        for i in range(start_idx, len(df)):
            curr_close = close.iloc[i]
            prev_close = close.iloc[i - 1]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            if i == start_idx:
                # Initialize first value
                if curr_close > curr_upper:
                    supertrend.iloc[i] = curr_lower
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = curr_upper
                    direction.iloc[i] = -1
            else:
                prev_supertrend = supertrend.iloc[i - 1]
                prev_direction = direction.iloc[i - 1]

                if prev_direction == 1:  # Previous was bullish
                    if curr_close < prev_supertrend:
                        # Flip to bearish
                        supertrend.iloc[i] = curr_upper
                        direction.iloc[i] = -1
                    else:
                        # Stay bullish, use lower band (but not lower than previous)
                        supertrend.iloc[i] = max(curr_lower, prev_supertrend)
                        direction.iloc[i] = 1
                else:  # Previous was bearish
                    if curr_close > prev_supertrend:
                        # Flip to bullish
                        supertrend.iloc[i] = curr_lower
                        direction.iloc[i] = 1
                    else:
                        # Stay bearish, use upper band (but not higher than previous)
                        supertrend.iloc[i] = min(curr_upper, prev_supertrend)
                        direction.iloc[i] = -1

        result = df.copy()
        result['supertrend'] = supertrend
        result['direction'] = direction
        return result

    def execute_strategy(
        self,
        df: pd.DataFrame,
        strategy_params: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute SuperTrend strategy deterministically.

        Rules:
        - Signal at time T
        - Entry at T+1 Open
        - Exit when direction flips
        """
        trades: List[Trade] = []
        equity = [100000.0]  # Starting capital
        position: Optional[Trade] = None

        atr_period = strategy_params.get('atr_period', 10)
        multiplier = strategy_params.get('multiplier', 3.0)

        # Calculate indicator
        df = self.calculate_supertrend(df, atr_period, multiplier)

        for i in range(1, len(df)):
            current_date = df.index[i].strftime('%Y-%m-%d')
            prev_direction = df['direction'].iloc[i - 1]
            curr_direction = df['direction'].iloc[i]
            entry_price = df['Open'].iloc[i]  # T+1 Open

            # Check for exit first
            if position is not None:
                # Exit on direction flip
                should_exit = (
                    (position.direction == 'LONG' and curr_direction == -1) or
                    (position.direction == 'SHORT' and curr_direction == 1)
                )
                if should_exit:
                    exit_price = entry_price
                    if position.direction == 'LONG':
                        pnl = exit_price - position.entry_price
                    else:
                        pnl = position.entry_price - exit_price

                    completed_trade = Trade(
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=current_date,
                        exit_price=exit_price,
                        direction=position.direction,
                        pnl=pnl
                    )
                    trades.append(completed_trade)
                    equity.append(equity[-1] + pnl * 50)  # Lot size multiplier
                    position = None

            # Check for entry (only if flat)
            if position is None and not pd.isna(prev_direction):
                if prev_direction == 1 and curr_direction == 1:
                    # Long entry signal
                    position = Trade(
                        entry_date=current_date,
                        entry_price=entry_price,
                        exit_date=None,
                        exit_price=None,
                        direction='LONG'
                    )
                elif prev_direction == -1 and curr_direction == -1:
                    # Short entry signal
                    position = Trade(
                        entry_date=current_date,
                        entry_price=entry_price,
                        exit_date=None,
                        exit_price=None,
                        direction='SHORT'
                    )

        # Close any open position at end
        if position is not None:
            exit_price = df['Close'].iloc[-1]
            if position.direction == 'LONG':
                pnl = exit_price - position.entry_price
            else:
                pnl = position.entry_price - exit_price

            completed_trade = Trade(
                entry_date=position.entry_date,
                entry_price=position.entry_price,
                exit_date=df.index[-1].strftime('%Y-%m-%d'),
                exit_price=exit_price,
                direction=position.direction,
                pnl=pnl
            )
            trades.append(completed_trade)
            equity.append(equity[-1] + pnl * 50)

        # Build equity curve DataFrame
        equity_df = pd.DataFrame({
            'equity': equity
        })

        # Calculate professional metrics
        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        # Calculate returns for Sharpe/Sortino
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # Max Drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # CAGR (Compound Annual Growth Rate)
        initial_equity = equity[0]
        final_equity = equity[-1]
        n_days = len(df) if len(df) > 0 else 1
        years = n_days / 252
        if years > 0 and initial_equity > 0:
            cagr = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0
        
        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average trade
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_trade = total_pnl / len(trades) if trades else 0
        
        # Expectancy
        win_rate = len(winning_trades) / len(trades) if trades else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'final_equity': equity[-1],
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'cagr': round(cagr, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'expectancy': round(expectancy, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        }

        # Generate execution hash for audit
        import hashlib
        hash_input = f"{len(trades)}_{total_pnl}_{equity[-1]}_{sharpe_ratio}"
        execution_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return ExecutionResult(
            success=True,
            error_message=None,
            trades=trades,
            equity_curve=equity_df,
            metrics=metrics,
            execution_hash=execution_hash
        )
