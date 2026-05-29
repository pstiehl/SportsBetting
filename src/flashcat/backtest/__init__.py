"""Backtest harness: simulate $100 flat bets, score sources by Brier + ROI."""

from .grader import brier_score, simulate_bet, simulate_bets, build_scoreboard
from .runner import run_backtest

__all__ = ["brier_score", "simulate_bet", "simulate_bets", "build_scoreboard", "run_backtest"]
