"""
Week 01 - FFT Adaptive-Spread Market Making Strategy

Cleaned version for GitHub archive.

Original idea:
- trade AAPL and MSFT with a simple market-making strategy;
- track recent mid-prices in a rolling window;
- use FFT energy as a rough proxy for short-term price instability;
- widen the required spread when the signal was noisy;
- adjust reservation price using inventory and order book imbalance;
- flatten positions at the end of the session.

This script is a cleaned archival version of the Week 01 strategy iteration.
It removes private credentials and local configuration details. It requires the
SHIFT trading environment.
"""

from __future__ import annotations

import os
from datetime import datetime, time as dt_time
from time import sleep
from typing import Dict, List

import numpy as np

try:
    import shift
    HAS_SHIFT = True
except ImportError:
    HAS_SHIFT = False


# ============================================================
# Configuration
# ============================================================

TICKERS = ["AAPL", "MSFT"]

END_TIME = dt_time(15, 50, 0)
SLEEP_SEC = 1.0

BASE_ORDER_SIZE = 1
MAX_INVENTORY_LOTS = 5

BASE_SPREAD = 0.02
MAX_SPREAD = 0.25
FFT_WINDOW = 64

IMBALANCE_COEF = 0.02
INVENTORY_GAMMA = 0.10

JUMP_FILTER = 0.08
DEFAULT_SIGMA = 0.01


# ============================================================
# Utility functions
# ============================================================

def cancel_orders(trader, ticker: str) -> None:
    """Cancel all resting orders for a ticker."""

    orders_to_cancel = [
        order for order in list(trader.get_waiting_list()) if order.symbol == ticker
    ]

    for order in orders_to_cancel:
        trader.submit_cancellation(order)


def close_positions(trader, ticker: str) -> None:
    """Close remaining long or short shares for a ticker."""

    item = trader.get_portfolio_item(ticker)
    long_shares = item.get_long_shares()
    short_shares = item.get_short_shares()

    if long_shares > 0:
        lots = int(long_shares // 100)
        if lots > 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_SELL, ticker, lots))
            sleep(1)

    if short_shares > 0:
        lots = int(short_shares // 100)
        if lots > 0:
            trader.submit_order(shift.Order(shift.Order.Type.MARKET_BUY, ticker, lots))
            sleep(1)


def fft_min_spread(mid_prices: List[float], base_spread: float = BASE_SPREAD) -> float:
    """
    Estimate a minimum spread using high-frequency FFT energy.

    A larger high-frequency component indicates noisier short-term movement, so the
    strategy requires a wider spread before quoting.
    """

    prices = np.array(mid_prices, dtype=float)

    if len(prices) < 8:
        return base_spread

    signal = prices - np.mean(prices)
    fft_values = np.fft.rfft(signal)
    magnitudes = np.abs(fft_values)

    high_frequency_energy = np.sum(magnitudes[len(magnitudes) // 2 :])
    adjusted_spread = base_spread * (1 + high_frequency_energy / (len(prices) + 1e-9))

    return float(max(base_spread, adjusted_spread))


def get_inventory_lots(trader, ticker: str) -> float:
    item = trader.get_portfolio_item(ticker)
    net_shares = item.get_long_shares() - item.get_short_shares()
    return net_shares / 100.0


def order_book_imbalance(bid_size: float, ask_size: float) -> float:
    total_size = bid_size + ask_size
    if total_size <= 0:
        return 0.0
    return (bid_size - ask_size) / total_size


# ============================================================
# Strategy logic
# ============================================================

def strategy_step(trader, ticker: str, state: Dict[str, dict]) -> None:
    """Run one strategy update for a ticker."""

    if ticker not in state:
        state[ticker] = {"recent_mids": []}

    recent_mids = state[ticker]["recent_mids"]

    best_price = trader.get_best_price(ticker)
    best_bid = best_price.get_bid_price()
    best_ask = best_price.get_ask_price()

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return

    spread = best_ask - best_bid
    midprice = 0.5 * (best_bid + best_ask)

    bid_size = best_price.get_bid_size()
    ask_size = best_price.get_ask_size()

    if bid_size + ask_size <= 0:
        return

    imbalance = order_book_imbalance(bid_size, ask_size)

    recent_mids.append(midprice)
    if len(recent_mids) > FFT_WINDOW:
        recent_mids.pop(0)

    min_spread = fft_min_spread(recent_mids, base_spread=BASE_SPREAD)

    if spread < min_spread or spread > MAX_SPREAD:
        cancel_orders(trader, ticker)
        return

    if len(recent_mids) >= 5:
        sigma = float(np.std(np.diff(recent_mids)))
    else:
        sigma = DEFAULT_SIGMA

    if len(recent_mids) >= 2 and abs(recent_mids[-1] - recent_mids[-2]) > JUMP_FILTER:
        cancel_orders(trader, ticker)
        return

    inventory_lots = get_inventory_lots(trader, ticker)

    reservation_price = (
        midprice
        - INVENTORY_GAMMA * inventory_lots * (sigma**2)
        + IMBALANCE_COEF * imbalance
    )

    target_half_spread = max(min_spread / 2.0, 1.5 * sigma)

    bid_quote = round(reservation_price - target_half_spread, 2)
    ask_quote = round(reservation_price + target_half_spread, 2)

    bid_quote = min(bid_quote, best_bid)
    ask_quote = max(ask_quote, best_ask)

    buy_size = BASE_ORDER_SIZE
    sell_size = BASE_ORDER_SIZE

    if inventory_lots >= MAX_INVENTORY_LOTS:
        buy_size = 0
        sell_size = BASE_ORDER_SIZE + 1
    elif inventory_lots <= -MAX_INVENTORY_LOTS:
        sell_size = 0
        buy_size = BASE_ORDER_SIZE + 1
    else:
        if imbalance > 0.35:
            buy_size += 1
        elif imbalance < -0.35:
            sell_size += 1

    cancel_orders(trader, ticker)

    if buy_size > 0:
        order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, int(buy_size))
        order.price = bid_quote
        trader.submit_order(order)

    if sell_size > 0:
        order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, int(sell_size))
        order.price = ask_quote
        trader.submit_order(order)


def main(trader) -> None:
    current = trader.get_last_trade_time()
    end_time = datetime.combine(current.date(), END_TIME)

    while trader.get_last_trade_time() < current:
        sleep(1)

    initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
    state: Dict[str, dict] = {}

    print("Week 01 FFT adaptive-spread strategy started.", flush=True)

    while trader.get_last_trade_time() < end_time:
        for ticker in TICKERS:
            strategy_step(trader, ticker, state)
        sleep(SLEEP_SEC)

    for ticker in TICKERS:
        cancel_orders(trader, ticker)
        close_positions(trader, ticker)

    final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_pl
    final_bp = trader.get_portfolio_summary().get_total_bp()

    print("Week 01 strategy stopped.", flush=True)
    print(f"final BP: {final_bp:.2f}", flush=True)
    print(f"final PnL: {final_pl:+.2f}", flush=True)


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
        sleep(2)
        main(trader)
