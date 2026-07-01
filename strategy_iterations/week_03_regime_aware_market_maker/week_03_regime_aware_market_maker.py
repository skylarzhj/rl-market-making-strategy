"""
Week 03 - Regime-Aware Market Making Strategy

Cleaned version for GitHub archive.

Original idea:
- trade a basket of tickers with a market-making strategy;
- estimate fair value using microprice;
- adjust quotes based on inventory;
- track short-term volatility with a GARCH(1,1)-style update;
- classify market regime using downside deviation and Sortino-style features;
- use the inferred regime to widen quotes and reduce inventory in weaker markets.

This script is a cleaned archival version of the Week 03 strategy iteration.
It removes private credentials and local configuration details. It requires the
SHIFT trading environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from time import sleep
from typing import List, Optional

import numpy as np

try:
    import shift
    HAS_SHIFT = True
except ImportError:
    HAS_SHIFT = False


# ============================================================
# Configuration
# ============================================================

TICKERS = [
    "CAR", "BGS", "COLM", "CROX", "ENR", "HELE", "JACK",
    "PZZA", "SAM", "SHAK", "SHOO", "TXRH", "WDFC", "WING", "YETI",
]

END_TIME = dt_time(15, 50, 0)
HARD_FLATTEN_TIME = dt_time(15, 49, 0)

BASE_HALF_SPREAD = 0.003
INVENTORY_SKEW = 0.015
FIXED_COST = 0.0003

GARCH_ALPHA0 = 1e-7
GARCH_ALPHA1 = 0.08
GARCH_BETA1 = 0.90
VOL_FLOOR = 1e-6

HL_DD = 10
HL_SORTINO_20 = 20
HL_SORTINO_60 = 60
WARMUP_SAMPLES = 30
JUMP_PENALTY = 50.0

BULL_MAX_INVENTORY = 10
BEAR_MAX_INVENTORY = 3
BEAR_SPREAD_MULTIPLIER = 2.0


# ============================================================
# Data classes
# ============================================================

@dataclass
class TradingStats:
    submitted: int = 0
    executions: int = 0


@dataclass
class RegimeState:
    """
    Online two-state regime classifier.

    Features:
    - downside deviation with half-life 10
    - Sortino-style ratio with half-life 20
    - Sortino-style ratio with half-life 60

    State 0 is treated as normal/bullish.
    State 1 is treated as defensive/bearish.
    """

    ewm_return_20: float = 0.0
    ewm_return_60: float = 0.0
    ewm_downside_10: float = 0.0
    ewm_downside_20: float = 0.0
    ewm_downside_60: float = 0.0

    feature_window: List[np.ndarray] = field(default_factory=list)
    max_window: int = 200

    regime: int = 0
    previous_regime: int = 0
    regime_shift_count: int = 0

    centroid_bull: Optional[np.ndarray] = None
    centroid_bear: Optional[np.ndarray] = None
    initialized: bool = False

    alpha_10: float = field(init=False)
    alpha_20: float = field(init=False)
    alpha_60: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_10 = 1 - 2 ** (-1.0 / HL_DD)
        self.alpha_20 = 1 - 2 ** (-1.0 / HL_SORTINO_20)
        self.alpha_60 = 1 - 2 ** (-1.0 / HL_SORTINO_60)

    def update_ewm_statistics(self, ret: float) -> None:
        downside_squared = ret**2 if ret < 0 else 0.0

        self.ewm_downside_10 = (
            (1 - self.alpha_10) * self.ewm_downside_10
            + self.alpha_10 * downside_squared
        )
        self.ewm_downside_20 = (
            (1 - self.alpha_20) * self.ewm_downside_20
            + self.alpha_20 * downside_squared
        )
        self.ewm_downside_60 = (
            (1 - self.alpha_60) * self.ewm_downside_60
            + self.alpha_60 * downside_squared
        )

        self.ewm_return_20 = (1 - self.alpha_20) * self.ewm_return_20 + self.alpha_20 * ret
        self.ewm_return_60 = (1 - self.alpha_60) * self.ewm_return_60 + self.alpha_60 * ret

    def features(self) -> np.ndarray:
        dd10 = max(np.sqrt(self.ewm_downside_10), 1e-8)
        dd20 = max(np.sqrt(self.ewm_downside_20), 1e-8)
        dd60 = max(np.sqrt(self.ewm_downside_60), 1e-8)

        sortino_20 = self.ewm_return_20 / dd20
        sortino_60 = self.ewm_return_60 / dd60

        return np.array([dd10, sortino_20, sortino_60], dtype=float)

    def _initialize_centroids(self) -> None:
        data = np.array(self.feature_window)
        midpoint = len(data) // 2

        c0 = data[:midpoint].mean(axis=0)
        c1 = data[midpoint:].mean(axis=0)

        for _ in range(5):
            labels = np.array(
                [0 if np.linalg.norm(x - c0) < np.linalg.norm(x - c1) else 1 for x in data]
            )

            if labels.sum() == 0 or labels.sum() == len(labels):
                break

            c0 = data[labels == 0].mean(axis=0)
            c1 = data[labels == 1].mean(axis=0)

        # Lower downside deviation is treated as the more stable/bullish centroid.
        if c0[0] < c1[0]:
            self.centroid_bull = c0
            self.centroid_bear = c1
        else:
            self.centroid_bull = c1
            self.centroid_bear = c0

        self.initialized = True

    def _regime_sequence_dp(self) -> List[int]:
        """
        Dynamic programming regime smoothing.

        Objective:
        minimize local distance-to-centroid loss plus a penalty for switching states.
        """

        data = np.array(self.feature_window)
        n_steps = len(data)
        n_states = 2
        centroids = [self.centroid_bull, self.centroid_bear]

        inf = float("inf")
        cost = [[inf] * n_states for _ in range(n_steps)]
        back = [[0] * n_states for _ in range(n_steps)]

        for state in range(n_states):
            cost[0][state] = 0.5 * np.sum((data[0] - centroids[state]) ** 2)

        for t in range(1, n_steps):
            for state in range(n_states):
                local_loss = 0.5 * np.sum((data[t] - centroids[state]) ** 2)

                best_prev_cost = inf
                best_prev_state = 0

                for prev_state in range(n_states):
                    jump_cost = JUMP_PENALTY if prev_state != state else 0.0
                    candidate_cost = cost[t - 1][prev_state] + jump_cost

                    if candidate_cost < best_prev_cost:
                        best_prev_cost = candidate_cost
                        best_prev_state = prev_state

                cost[t][state] = local_loss + best_prev_cost
                back[t][state] = best_prev_state

        sequence = [0] * n_steps
        sequence[-1] = int(np.argmin(cost[-1]))

        for t in range(n_steps - 2, -1, -1):
            sequence[t] = back[t + 1][sequence[t + 1]]

        return sequence

    def update_regime(self, ret: float) -> int:
        self.update_ewm_statistics(ret)
        feature_vector = self.features()

        self.feature_window.append(feature_vector)
        if len(self.feature_window) > self.max_window:
            self.feature_window.pop(0)

        if len(self.feature_window) < WARMUP_SAMPLES:
            return self.regime

        if not self.initialized:
            self._initialize_centroids()

        if len(self.feature_window) % 10 == 0:
            self._initialize_centroids()

        sequence = self._regime_sequence_dp()
        new_regime = sequence[-1]

        self.previous_regime = self.regime
        self.regime = new_regime

        if new_regime != self.previous_regime:
            self.regime_shift_count += 1

        return self.regime


@dataclass
class SymbolState:
    prev_mid: Optional[float] = None
    garch_var: float = VOL_FLOOR
    last_bid_quote: Optional[float] = None
    last_ask_quote: Optional[float] = None
    had_quotes: bool = False
    samples: int = 0
    last_fill_count: int = 0
    regime_state: RegimeState = field(default_factory=RegimeState)


# ============================================================
# Utility functions
# ============================================================

def debug(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def round_down_cent(price: float) -> float:
    return round(np.floor(price * 100.0) / 100.0, 2)


def round_up_cent(price: float) -> float:
    return round(np.ceil(price * 100.0) / 100.0, 2)


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Microprice fair-value estimate using top-of-book size imbalance."""

    total_size = bid_size + ask_size
    if total_size <= 0:
        return 0.5 * (bid + ask)

    return (bid * ask_size + ask * bid_size) / total_size


def clear_quotes(trader, ticker: str, state: SymbolState) -> None:
    """Cancel active quotes for one ticker."""

    if state.had_quotes:
        for order in list(trader.get_waiting_list()):
            if order.symbol == ticker:
                trader.submit_cancellation(order)

    state.last_bid_quote = None
    state.last_ask_quote = None
    state.had_quotes = False


# ============================================================
# Core market-making logic
# ============================================================

def step_symbol(trader, ticker: str, state: SymbolState, stats: TradingStats) -> None:
    best = trader.get_best_price(ticker)

    bid = best.get_bid_price()
    ask = best.get_ask_price()
    bid_size = best.get_bid_size()
    ask_size = best.get_ask_size()

    if bid <= 0.01 or ask <= 0.01 or ask <= bid:
        return

    mid = 0.5 * (bid + ask)

    if state.prev_mid is None:
        state.prev_mid = mid
        return

    ret = np.log(mid / state.prev_mid)
    state.prev_mid = mid
    state.samples += 1

    state.garch_var = max(
        GARCH_ALPHA0 + GARCH_ALPHA1 * (ret**2) + GARCH_BETA1 * state.garch_var,
        VOL_FLOOR,
    )

    regime = state.regime_state.update_regime(ret)

    if regime == 0:
        max_inventory = BULL_MAX_INVENTORY
        half_spread = BASE_HALF_SPREAD
    else:
        max_inventory = BEAR_MAX_INVENTORY
        half_spread = BASE_HALF_SPREAD * BEAR_SPREAD_MULTIPLIER

    portfolio = trader.get_portfolio_item(ticker)
    inventory = int((portfolio.get_long_shares() - portfolio.get_short_shares()) / 100)

    fair_price = microprice(bid, ask, bid_size, ask_size)
    reservation_price = fair_price - INVENTORY_SKEW * inventory

    bid_quote = round_down_cent(min(bid, reservation_price - half_spread))
    ask_quote = round_up_cent(max(ask, reservation_price + half_spread))

    bid_edge = fair_price - bid_quote
    ask_edge = ask_quote - fair_price

    if bid_edge <= FIXED_COST and ask_edge <= FIXED_COST:
        return

    bid_changed = abs((state.last_bid_quote or 0) - bid_quote) >= 0.01
    ask_changed = abs((state.last_ask_quote or 0) - ask_quote) >= 0.01

    if not state.had_quotes or bid_changed or ask_changed:
        clear_quotes(trader, ticker, state)

        if inventory < max_inventory:
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, 1, bid_quote)
            if trader.submit_order(order):
                stats.submitted += 1

        if inventory > -max_inventory:
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, 1, ask_quote)
            if trader.submit_order(order):
                stats.submitted += 1

        state.last_bid_quote = bid_quote
        state.last_ask_quote = ask_quote
        state.had_quotes = True


def flatten_positions(trader, states: dict[str, SymbolState]) -> None:
    """Cancel open orders and close remaining positions near the end of the session."""

    debug("Flattening positions and cancelling open orders.")

    for order in list(trader.get_waiting_list()):
        trader.submit_cancellation(order)

    for ticker, state in states.items():
        portfolio = trader.get_portfolio_item(ticker)
        shares = portfolio.get_long_shares() - portfolio.get_short_shares()

        if shares > 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, ticker, abs(shares) // 100))
        elif shares < 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, ticker, abs(shares) // 100))

        state.had_quotes = False


# ============================================================
# Main loop
# ============================================================

def main(trader) -> None:
    while trader.get_last_trade_time().year < 2000:
        debug("Waiting for market start...")
        sleep(1)

    now = trader.get_last_trade_time()
    session_day = now.date()
    end_dt = datetime.combine(session_day, END_TIME)
    flatten_dt = datetime.combine(session_day, HARD_FLATTEN_TIME)

    states = {ticker: SymbolState() for ticker in TICKERS}
    stats = TradingStats()

    for ticker in TICKERS:
        states[ticker].last_fill_count = len(trader.get_executed_orders(ticker))

    debug("Trading activated: regime-aware market maker.")

    while trader.get_last_trade_time() < end_dt:
        now = trader.get_last_trade_time()

        if now >= flatten_dt:
            flatten_positions(trader, states)
            break

        for ticker in TICKERS:
            current_fills = len(trader.get_executed_orders(ticker))
            new_fills = current_fills - states[ticker].last_fill_count

            if new_fills > 0:
                stats.executions += new_fills
                states[ticker].last_fill_count = current_fills

                regime_label = "BULL" if states[ticker].regime_state.regime == 0 else "BEAR"
                debug(
                    f"Fill: {ticker} | regime={regime_label} | "
                    f"total_fills={stats.executions} | "
                    f"regime_shifts={states[ticker].regime_state.regime_shift_count}"
                )

            step_symbol(trader, ticker, states[ticker], stats)

        sleep(0.1)

    debug(f"Session ended. submitted={stats.submitted}, executions={stats.executions}")


if __name__ == "__main__":
    if not HAS_SHIFT:
        raise ImportError("The SHIFT package is required to run this strategy.")

    username = os.getenv("SHIFT_USERNAME")
    password = os.getenv("SHIFT_PASSWORD")
    config_file = os.getenv("SHIFT_CONFIG", "initiator.cfg")

    if not username or not password:
        raise EnvironmentError(
            "SHIFT_USERNAME and SHIFT_PASSWORD must be set as environment variables."
        )

    with shift.Trader(username) as trader:
        trader.connect(config_file, password)
        trader.sub_all_order_book()
        main(trader)
