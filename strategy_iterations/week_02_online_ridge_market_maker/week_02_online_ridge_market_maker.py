"""
Week 02 - Online Ridge Market Making Strategy

Cleaned version for GitHub archive.

Original idea:
- trade CSCO, MSFT, and NVDA with a statistical market-making framework;
- build short-horizon features from microprice, spread, order book imbalance,
  fast/slow return signals, volatility, and inventory;
- train an online ridge regression model using recent observations;
- use the predicted short-term return to adjust quotes and order sizes;
- apply inventory limits and end-of-session flattening.

This script is a cleaned archival version of the Week 02 strategy iteration.
It removes private credentials and local configuration details. It requires the
SHIFT trading environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from time import sleep
from typing import List, Optional, Tuple

import numpy as np

try:
    import shift
    HAS_SHIFT = True
except ImportError:
    HAS_SHIFT = False


# ============================================================
# Configuration
# ============================================================

TICKERS = ["CSCO", "MSFT", "NVDA"]

END_TIME = dt_time(15, 50, 0)
SOFT_FLATTEN_TIME = dt_time(15, 47, 0)
HARD_FLATTEN_TIME = dt_time(15, 49, 20)

SLEEP_SEC = 0.05
CANCEL_PAUSE_SEC = 0.0

LOTS_PER_ORDER_MIN = 1
LOTS_PER_ORDER_MAX = 2
BASE_MAX_INVENTORY = 4
BASE_MAX_GROSS_INVENTORY = 4

QUOTE_MAX_AGE_SEC = 1.0
REPRICE_THRESHOLD = 0.01
IMPROVE_ONE_TICK_IF_POSSIBLE = True

EWMA_LAMBDA = 0.94
VOL_FLOOR = 1e-6
VOL_CEILING = 0.20

FEATURE_WINDOW = 180
RIDGE_LAMBDA = 1e-3
MIN_TRAIN_SAMPLES = 30
TRAIN_EVERY_N = 20

EDGE_TO_QUOTE = 0.0080
EDGE_TO_ONE_SIDE = 0.015
EDGE_TO_TAKE = 0.0250

BASE_HALF_SPREAD = 0.015
VOL_HALF_SPREAD_MULT = 3.0
INVENTORY_HALF_SPREAD_MULT = 0.004
ADVERSE_SELECTION_MULT = 0.30

INVENTORY_SKEW = 0.02
LATE_DAY_INVENTORY_SKEW = 0.04

FILL_DECAY = 80.0
FIXED_COST_PER_SHARE = 0.0005
VOL_COST_MULT = 0.30
INVENTORY_COST_MULT = 0.0015

MIN_OBSERVED_SPREAD = 0.02
MAX_OBSERVED_SPREAD = 0.15
MIN_DEPTH = 1
MAX_VOL_PRICE = 0.06

TAKER_COOLDOWN_SEC = 3.0

PRINT_DEBUG = True


# ============================================================
# Data classes
# ============================================================

@dataclass
class SymbolState:
    prev_mid: Optional[float] = None
    ewma_var: float = VOL_FLOOR
    imbalance_ema: float = 0.0
    fast_ret_ema: float = 0.0
    slow_ret_ema: float = 0.0

    samples: int = 0
    X_hist: List[np.ndarray] = field(default_factory=list)
    y_hist: List[float] = field(default_factory=list)
    beta: Optional[np.ndarray] = None
    last_x: Optional[np.ndarray] = None

    last_bid_quote: Optional[float] = None
    last_ask_quote: Optional[float] = None
    last_quote_time: Optional[datetime] = None
    last_taker_time: Optional[datetime] = None
    had_quotes: bool = False


# ============================================================
# Utility functions
# ============================================================

def debug(message: str) -> None:
    if PRINT_DEBUG:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def round_down_cent(price: float) -> float:
    return round(np.floor(price * 100.0) / 100.0, 2)


def round_up_cent(price: float) -> float:
    return round(np.ceil(price * 100.0) / 100.0, 2)


def seconds_since(now: datetime, previous: Optional[datetime]) -> float:
    if previous is None:
        return float("inf")
    return (now - previous).total_seconds()


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total <= 0:
        return 0.5 * (bid + ask)
    return (bid * ask_size + ask * bid_size) / total


def imbalance(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


def get_inventory_lots(trader, ticker: str) -> int:
    item = trader.get_portfolio_item(ticker)
    return int((item.get_long_shares() - item.get_short_shares()) / 100)


def get_gross_inventory_lots(trader, tickers: List[str]) -> int:
    return int(sum(abs(get_inventory_lots(trader, ticker)) for ticker in tickers))


def dynamic_inventory_limit(now: datetime, session_day) -> int:
    soft = datetime.combine(session_day, SOFT_FLATTEN_TIME)
    hard = datetime.combine(session_day, HARD_FLATTEN_TIME)

    if now < soft:
        return BASE_MAX_INVENTORY

    total = max((hard - soft).total_seconds(), 1.0)
    left = max((hard - now).total_seconds(), 0.0)
    return max(1, int(np.ceil(BASE_MAX_INVENTORY * left / total)))


def cancel_orders_for_ticker(trader, ticker: str) -> None:
    for order in list(trader.get_waiting_list()):
        if order.symbol == ticker:
            trader.submit_cancellation(order)
    sleep(CANCEL_PAUSE_SEC)


def clear_quotes(trader, ticker: str, state: SymbolState) -> None:
    if state.had_quotes:
        cancel_orders_for_ticker(trader, ticker)

    state.last_bid_quote = None
    state.last_ask_quote = None
    state.had_quotes = False


def close_positions(trader, ticker: str) -> None:
    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()

    if long_shares > 0:
        trader.submit_order(
            shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares // 100))
        )

    if short_shares > 0:
        trader.submit_order(
            shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares // 100))
        )


def full_cleanup(trader, tickers: List[str]) -> None:
    debug("Cleaning open orders and positions.")

    for order in list(trader.get_waiting_list()):
        trader.submit_cancellation(order)

    sleep(0.2)

    for ticker in tickers:
        close_positions(trader, ticker)

    sleep(0.5)
    debug("Cleanup complete.")


# ============================================================
# Online regression
# ============================================================

def build_feature_vector(
    mid: float,
    micro: float,
    spread: float,
    current_imbalance: float,
    imbalance_ema: float,
    fast_ret_ema: float,
    slow_ret_ema: float,
    vol_price: float,
    inventory_lots: int,
) -> np.ndarray:
    return np.array(
        [
            1.0,
            micro - mid,
            spread,
            current_imbalance,
            imbalance_ema,
            fast_ret_ema,
            slow_ret_ema,
            vol_price,
            inventory_lots,
        ],
        dtype=float,
    )


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    n_features = X.shape[1]
    reg = ridge_lambda * np.eye(n_features)
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def maybe_update_model(state: SymbolState) -> None:
    if len(state.X_hist) < MIN_TRAIN_SAMPLES or state.samples % TRAIN_EVERY_N != 0:
        return

    X = np.vstack(state.X_hist)
    y = np.array(state.y_hist, dtype=float)

    try:
        state.beta = fit_ridge(X, y, RIDGE_LAMBDA)
    except np.linalg.LinAlgError:
        state.beta = None


def predict_next_return(state: SymbolState, x: np.ndarray) -> float:
    if state.beta is None:
        # Hand-tuned fallback before enough data accumulates.
        return float(
            0.80 * x[1]
            + 0.0020 * x[3]
            + 0.0030 * x[4]
            + 5.0 * x[5]
            - 0.20 * x[6]
            - 0.0008 * x[7]
        )

    return float(state.beta @ x)


def process_new_observation(state: SymbolState, mid: float, x_now: np.ndarray) -> None:
    if state.last_x is not None and state.prev_mid is not None:
        y = mid - state.prev_mid
        state.X_hist.append(state.last_x)
        state.y_hist.append(float(y))

        if len(state.X_hist) > FEATURE_WINDOW:
            state.X_hist.pop(0)
            state.y_hist.pop(0)

    state.last_x = x_now


# ============================================================
# Quoting logic
# ============================================================

def expected_fill_prob(distance_from_touch: float) -> float:
    return float(np.exp(-FILL_DECAY * max(distance_from_touch, 0.0)))


def expected_trade_value(
    quoted_edge_per_share: float,
    fill_prob: float,
    vol_price: float,
    inventory_lots: int,
) -> float:
    cost = (
        FIXED_COST_PER_SHARE
        + VOL_COST_MULT * vol_price
        + INVENTORY_COST_MULT * abs(inventory_lots)
    )
    return fill_prob * quoted_edge_per_share - cost


def compute_quotes(
    best_bid: float,
    best_ask: float,
    fair_price: float,
    observed_spread: float,
    vol_price: float,
    predicted_ret: float,
    inventory_lots: int,
    now: datetime,
    session_day,
) -> Tuple[Optional[float], Optional[float]]:
    half_spread = max(
        BASE_HALF_SPREAD,
        0.5 * observed_spread,
        VOL_HALF_SPREAD_MULT * vol_price,
        INVENTORY_HALF_SPREAD_MULT * abs(inventory_lots),
        ADVERSE_SELECTION_MULT * abs(predicted_ret),
    )

    mid = 0.5 * (best_bid + best_ask)
    reservation = mid - INVENTORY_SKEW * inventory_lots

    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        reservation -= LATE_DAY_INVENTORY_SKEW * inventory_lots

    raw_bid = reservation - half_spread
    raw_ask = reservation + half_spread

    if IMPROVE_ONE_TICK_IF_POSSIBLE and observed_spread >= 0.02:
        bid_quote = min(best_bid + 0.01, round_down_cent(raw_bid))
        ask_quote = max(best_ask - 0.01, round_up_cent(raw_ask))
    else:
        bid_quote = min(best_bid, round_down_cent(raw_bid))
        ask_quote = max(best_ask, round_up_cent(raw_ask))

    bid_quote = min(bid_quote, round(best_ask - 0.01, 2))
    ask_quote = max(ask_quote, round(best_bid + 0.01, 2))

    if bid_quote >= ask_quote:
        bid_quote = round(best_bid, 2)
        ask_quote = round(best_ask, 2)

    bid_enabled = True
    ask_enabled = True

    if predicted_ret >= EDGE_TO_ONE_SIDE and inventory_lots <= 0:
        ask_enabled = False
    elif predicted_ret <= -EDGE_TO_ONE_SIDE and inventory_lots >= 0:
        bid_enabled = False

    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        if inventory_lots > 0:
            bid_enabled = False
            ask_quote = round(best_ask, 2)
        elif inventory_lots < 0:
            ask_enabled = False
            bid_quote = round(best_bid, 2)

    return bid_quote if bid_enabled else None, ask_quote if ask_enabled else None


def compute_quote_sizes(
    predicted_ret: float,
    vol_price: float,
    inventory_lots: int,
    inventory_limit: int,
    observed_spread: float,
    now: datetime,
    session_day,
) -> Tuple[int, int]:
    size = LOTS_PER_ORDER_MIN

    if abs(predicted_ret) >= EDGE_TO_ONE_SIDE:
        size += 1
    if observed_spread >= 0.03:
        size += 1
    if vol_price >= 0.03:
        size -= 1
    if now >= datetime.combine(session_day, SOFT_FLATTEN_TIME):
        size = 1

    size = int(np.clip(size, LOTS_PER_ORDER_MIN, LOTS_PER_ORDER_MAX))
    buy_size = size
    sell_size = size

    if predicted_ret > 0:
        buy_size = min(LOTS_PER_ORDER_MAX, buy_size + 1)
        sell_size = max(0, sell_size - 1)
    elif predicted_ret < 0:
        sell_size = min(LOTS_PER_ORDER_MAX, sell_size + 1)
        buy_size = max(0, buy_size - 1)

    if inventory_lots >= inventory_limit:
        buy_size = 0
        sell_size = max(1, sell_size)
    elif inventory_lots <= -inventory_limit:
        sell_size = 0
        buy_size = max(1, buy_size)

    return buy_size, sell_size


def should_requote(
    now: datetime,
    state: SymbolState,
    bid_quote: Optional[float],
    ask_quote: Optional[float],
) -> bool:
    if not state.had_quotes or state.last_quote_time is None:
        return True

    if seconds_since(now, state.last_quote_time) >= QUOTE_MAX_AGE_SEC:
        return True

    bid_changed = (
        bid_quote is not None
        and state.last_bid_quote is not None
        and abs(bid_quote - state.last_bid_quote) >= REPRICE_THRESHOLD
    )
    ask_changed = (
        ask_quote is not None
        and state.last_ask_quote is not None
        and abs(ask_quote - state.last_ask_quote) >= REPRICE_THRESHOLD
    )

    return bid_changed or ask_changed


# ============================================================
# Core trading step
# ============================================================

def step_symbol(trader, ticker: str, tickers: List[str], state: SymbolState, now: datetime, session_day) -> None:
    best = trader.get_best_price(ticker)

    best_bid = best.get_bid_price()
    best_ask = best.get_ask_price()
    bid_size = best.get_bid_size()
    ask_size = best.get_ask_size()

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        if state.samples % 20 == 0:
            debug(f"[skip] {ticker}: invalid book bid={best_bid}, ask={best_ask}")
        clear_quotes(trader, ticker, state)
        return

    if bid_size < MIN_DEPTH or ask_size < MIN_DEPTH:
        clear_quotes(trader, ticker, state)
        return

    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    if spread < MIN_OBSERVED_SPREAD or spread > MAX_OBSERVED_SPREAD:
        clear_quotes(trader, ticker, state)
        return

    micro = microprice(best_bid, best_ask, bid_size, ask_size)
    imb_now = imbalance(bid_size, ask_size)

    if state.prev_mid is None:
        state.prev_mid = mid
        return

    ret = (mid - state.prev_mid) / max(state.prev_mid, 1e-8)
    state.ewma_var = EWMA_LAMBDA * state.ewma_var + (1.0 - EWMA_LAMBDA) * (ret**2)
    vol_price = mid * np.sqrt(max(state.ewma_var, VOL_FLOOR))

    if vol_price > VOL_CEILING:
        clear_quotes(trader, ticker, state)
        state.prev_mid = mid
        return

    state.imbalance_ema = 0.25 * imb_now + 0.75 * state.imbalance_ema
    state.fast_ret_ema = 0.30 * ret + 0.70 * state.fast_ret_ema
    state.slow_ret_ema = 0.08 * ret + 0.92 * state.slow_ret_ema
    state.samples += 1

    inventory_lots = get_inventory_lots(trader, ticker)
    gross_inventory_lots = get_gross_inventory_lots(trader, tickers)
    inventory_limit = dynamic_inventory_limit(now, session_day)

    x_now = build_feature_vector(
        mid,
        micro,
        spread,
        imb_now,
        state.imbalance_ema,
        state.fast_ret_ema,
        state.slow_ret_ema,
        vol_price,
        inventory_lots,
    )
    process_new_observation(state, mid, x_now)
    maybe_update_model(state)

    predicted_ret = predict_next_return(state, x_now)

    if gross_inventory_lots >= BASE_MAX_GROSS_INVENTORY and predicted_ret * inventory_lots >= 0:
        clear_quotes(trader, ticker, state)
        state.prev_mid = mid
        return

    taker_allowed = seconds_since(now, state.last_taker_time) >= TAKER_COOLDOWN_SEC
    if abs(predicted_ret) >= EDGE_TO_TAKE and taker_allowed:
        if (
            predicted_ret > 0
            and inventory_lots < inventory_limit
            and gross_inventory_lots < BASE_MAX_GROSS_INVENTORY
        ):
            clear_quotes(trader, ticker, state)
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, ticker, 1))
            state.last_taker_time = now
            debug(f"[taker buy] {ticker}: strong positive signal")
            state.prev_mid = mid
            return

        if (
            predicted_ret < 0
            and inventory_lots > -inventory_limit
            and gross_inventory_lots < BASE_MAX_GROSS_INVENTORY
        ):
            clear_quotes(trader, ticker, state)
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, ticker, 1))
            state.last_taker_time = now
            debug(f"[taker sell] {ticker}: strong negative signal")
            state.prev_mid = mid
            return

    if abs(predicted_ret) < EDGE_TO_QUOTE and spread <= 0.02:
        clear_quotes(trader, ticker, state)
        state.prev_mid = mid
        return

    fair_price = mid + predicted_ret
    bid_quote, ask_quote = compute_quotes(
        best_bid,
        best_ask,
        fair_price,
        spread,
        vol_price,
        predicted_ret,
        inventory_lots,
        now,
        session_day,
    )

    buy_size, sell_size = compute_quote_sizes(
        predicted_ret,
        vol_price,
        inventory_lots,
        inventory_limit,
        spread,
        now,
        session_day,
    )

    if should_requote(now, state, bid_quote, ask_quote):
        clear_quotes(trader, ticker, state)

        if bid_quote is not None and buy_size > 0:
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, int(buy_size))
            order.price = round(float(bid_quote), 2)
            trader.submit_order(order)
            debug(f"[bid] {ticker} px={bid_quote:.2f} size={buy_size}")

        if ask_quote is not None and sell_size > 0:
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, int(sell_size))
            order.price = round(float(ask_quote), 2)
            trader.submit_order(order)
            debug(f"[ask] {ticker} px={ask_quote:.2f} size={sell_size}")

        state.last_bid_quote = bid_quote
        state.last_ask_quote = ask_quote
        state.last_quote_time = now
        state.had_quotes = True

    state.prev_mid = mid


# ============================================================
# Main loop
# ============================================================

def main(trader) -> None:
    session_day = datetime.now().date()
    end_dt = datetime.combine(session_day, END_TIME)
    hard_flatten_dt = datetime.combine(session_day, HARD_FLATTEN_TIME)

    states = {ticker: SymbolState() for ticker in TICKERS}

    debug("Week 02 online ridge market maker started.")

    while datetime.now() < end_dt:
        now = datetime.now()

        if now >= hard_flatten_dt:
            debug("[flatten] hard flatten window reached")
            full_cleanup(trader, TICKERS)
            break

        for ticker in TICKERS:
            step_symbol(trader, ticker, TICKERS, states[ticker], now, session_day)

        sleep(SLEEP_SEC)

    full_cleanup(trader, TICKERS)
    debug("Week 02 strategy stopped.")


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
        sleep(2)
        trader.sub_all_order_book()
        sleep(5)
        full_cleanup(trader, TICKERS)
        main(trader)
