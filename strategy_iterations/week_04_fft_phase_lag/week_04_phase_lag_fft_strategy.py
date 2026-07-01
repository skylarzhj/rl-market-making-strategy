"""
Week 04 - FFT Phase-Lag Trading Strategy

Cleaned version for GitHub archive.

Original idea:
- collect high-frequency order book data for CS1, CS2, and CS3;
- use FFT to estimate short-term frequency components;
- estimate phase lag between leading tickers and CS2;
- predict CS2's near-future price;
- combine the prediction with order book imbalance to decide whether to place limit orders.

This script is a cleaned archival version of the Week 04 submission. It removes private
credentials and local configuration details. It requires the SHIFT trading environment.
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

try:
    import shift
    HAS_SHIFT = True
except ImportError:
    HAS_SHIFT = False


TICKERS = ["CS1", "CS2", "CS3"]
TARGET_TICKER = "CS2"

INTERVAL_SEC = 0.1
DEPTH = 5
WINDOW_SIZE = 500

ORDER_SIZE = 1
MAX_POSITION = 20
STOP_LOSS_PCT = 0.003

PREDICTION_THRESHOLD = 0.0005
OBI_THRESHOLD = 0.15

FFT_TOP_N = 3
FFT_UPDATE_EVERY_TICKS = 20
MIN_TAU_SEC = 1.0
MAX_TAU_SEC = 150.0

shared_buffers: Dict[str, deque] = {ticker: deque(maxlen=WINDOW_SIZE) for ticker in TICKERS}
buffer_lock = threading.Lock()


class BookCollector(threading.Thread):
    """Collects local order book snapshots into shared rolling buffers."""

    def __init__(self, trader):
        super().__init__(daemon=True, name="book-collector")
        self.trader = trader
        self._stop_event = threading.Event()
        self.tick_count = 0

    def _extract_book_snapshot(self, ticker: str) -> Optional[dict]:
        try:
            bids = self.trader.get_order_book(
                ticker, shift.OrderBookType.LOCAL_BID, max_level=DEPTH
            )
            asks = self.trader.get_order_book(
                ticker, shift.OrderBookType.LOCAL_ASK, max_level=DEPTH
            )
        except Exception:
            return None

        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 0.0
        if best_bid <= 0 or best_ask <= 0:
            return None

        bid_vol = sum(level.size for level in bids[:DEPTH])
        ask_vol = sum(level.size for level in asks[:DEPTH])
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        return {
            "wall_time": time.time(),
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "mid": (best_bid + best_ask) / 2,
            "bid": best_bid,
            "ask": best_ask,
            "obi": imbalance,
        }

    def run(self) -> None:
        print("[collector] started")
        while not self._stop_event.is_set():
            start = time.perf_counter()
            rows = {}

            for ticker in TICKERS:
                row = self._extract_book_snapshot(ticker)
                if row is not None:
                    rows[ticker] = row

            if rows:
                with buffer_lock:
                    for ticker, row in rows.items():
                        shared_buffers[ticker].append(row)

                self.tick_count += 1
                if self.tick_count % 100 == 0:
                    mids = {
                        t: shared_buffers[t][-1]["mid"] if shared_buffers[t] else 0
                        for t in TICKERS
                    }
                    print(
                        f"[collector {self.tick_count * INTERVAL_SEC:.1f}s] "
                        + " ".join(f"{t}={mids[t]:.3f}" for t in TICKERS)
                    )

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, INTERVAL_SEC - elapsed))

    def stop(self) -> None:
        self._stop_event.set()


def smooth_prices(prices: np.ndarray) -> np.ndarray:
    """Apply short-window Savitzky-Golay smoothing before FFT."""

    n = len(prices)
    window_length = min(21, n if n % 2 == 1 else n - 1)
    window_length = max(window_length, 5)
    return savgol_filter(prices, window_length=window_length, polyorder=2)


def fft_components(prices: np.ndarray, dt: float, top_n: int = 3) -> List[dict]:
    """Return the strongest positive-frequency FFT components."""

    clean_prices = np.nan_to_num(prices.astype(float), nan=np.nanmean(prices))
    denoised = smooth_prices(clean_prices)
    mean_value = float(np.mean(denoised))
    signal = denoised - mean_value

    n = len(signal)
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=dt)

    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_values[positive_mask]
    amplitudes = np.abs(positive_fft)

    if len(amplitudes) == 0:
        return []

    selected = np.argsort(amplitudes)[-top_n:][::-1]

    return [
        {
            "freq": float(positive_freqs[i]),
            "amplitude": float(amplitudes[i] / (n / 2)),
            "phase": float(np.angle(positive_fft[i])),
            "mean": mean_value,
        }
        for i in selected
    ]


def phase_lag_seconds(lead_component: dict, target_component: dict) -> float:
    """Estimate time lag from phase difference."""

    dphi = lead_component["phase"] - target_component["phase"]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return float(dphi / (2 * np.pi * target_component["freq"]))


def reconstruct_signal(components: List[dict], t: float) -> float:
    """Reconstruct a price estimate from FFT components."""

    if not components:
        return 0.0

    value = components[0]["mean"]
    for c in components:
        value += c["amplitude"] * np.cos(2 * np.pi * c["freq"] * t + c["phase"])
    return float(value)


class PhaseCache:
    """Background FFT cache so the main trading loop does not block."""

    def __init__(self, windows: Dict[str, deque], dt: float, top_n: int = 3):
        self.windows = windows
        self.dt = dt
        self.top_n = top_n
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.components: Dict[str, List[dict]] = {t: [] for t in TICKERS}
        self.tau_cs1 = 0.0
        self.tau_cs3 = 0.0
        self.best_tau = 0.0
        self.lead_ticker = TARGET_TICKER

        self._thread = threading.Thread(target=self._loop, daemon=True, name="phase-cache")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.dt * FFT_UPDATE_EVERY_TICKS)
            self._update()

    def _update(self) -> None:
        min_len = 200
        new_components = {}

        with buffer_lock:
            local_windows = {ticker: list(self.windows[ticker]) for ticker in TICKERS}

        for ticker in TICKERS:
            window = local_windows.get(ticker, [])
            if len(window) < min_len:
                return

            mids = np.array([row["mid"] for row in window], dtype=float)
            new_components[ticker] = fft_components(mids, self.dt, self.top_n)

        cs2_components = new_components.get(TARGET_TICKER, [])
        tau1 = 0.0
        tau3 = 0.0
        best_tau = 0.0
        lead = TARGET_TICKER

        if cs2_components:
            cs1_components = new_components.get("CS1", [])
            cs3_components = new_components.get("CS3", [])

            if cs1_components:
                tau1 = phase_lag_seconds(cs1_components[0], cs2_components[0])
            if cs3_components:
                tau3 = phase_lag_seconds(cs3_components[0], cs2_components[0])

            candidates = []
            if MIN_TAU_SEC <= tau1 <= MAX_TAU_SEC:
                candidates.append(("CS1", tau1))
            if MIN_TAU_SEC <= tau3 <= MAX_TAU_SEC:
                candidates.append(("CS3", tau3))

            if candidates:
                candidates.sort(key=lambda item: item[1])
                lead, best_tau = candidates[0]

        with self._lock:
            self.components = new_components
            self.tau_cs1 = tau1
            self.tau_cs3 = tau3
            self.best_tau = best_tau
            self.lead_ticker = lead

    def snapshot(self) -> Tuple[Dict[str, List[dict]], float, float, float, str]:
        with self._lock:
            return (
                dict(self.components),
                self.tau_cs1,
                self.tau_cs3,
                self.best_tau,
                self.lead_ticker,
            )

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)


class FFTPhaseLagStrategy:
    """Trading strategy using FFT phase lag and order book imbalance."""

    def __init__(self, target: str = TARGET_TICKER):
        self.target = target
        self.cache = PhaseCache(shared_buffers, dt=INTERVAL_SEC, top_n=FFT_TOP_N)
        self.collector = None

        self.has_position = False
        self.entry_price: Optional[float] = None
        self.tick_count = 0

    def attach_collector(self, collector: BookCollector) -> None:
        self.collector = collector

    def _best_price(self, trader) -> Optional[Tuple[float, float, float]]:
        bp = trader.get_best_price(self.target)
        bid = bp.get_bid_price()
        ask = bp.get_ask_price()

        if bid <= 0 or ask <= 0:
            return None

        return bid, ask, (bid + ask) / 2

    def _position(self, trader) -> int:
        return trader.get_portfolio_item(self.target).get_shares()

    def _latest_obi(self) -> float:
        with buffer_lock:
            window = shared_buffers.get(self.target)
            if not window:
                return 0.0
            return float(window[-1].get("obi", 0.0))

    def _predict_price(self, t_now: float) -> Tuple[Optional[float], float, str]:
        components, tau1, tau3, best_tau, lead = self.cache.snapshot()
        target_components = components.get(self.target, [])
        lead_components = components.get(lead, [])

        if not target_components:
            return None, 0.0, "no-fft"

        if best_tau > 0 and lead_components and lead != self.target:
            p_pred = target_components[0]["mean"]
            t_future = t_now + best_tau

            for lead_comp, target_comp in zip(lead_components, target_components):
                p_pred += lead_comp["amplitude"] * np.cos(
                    2 * np.pi * target_comp["freq"] * t_future + lead_comp["phase"]
                )

            return float(p_pred), best_tau, f"{lead}->{self.target} tau={best_tau:.1f}s"

        fallback = min([t for t in [tau1, tau3] if t > 1.0], default=10.0)
        p_pred = reconstruct_signal(target_components, t_now + fallback)

        return p_pred, fallback, f"{self.target}-only fallback={fallback:.0f}s"

    def _stop_loss_hit(self, mid: float) -> bool:
        return (
            self.has_position
            and self.entry_price is not None
            and mid < self.entry_price * (1 - STOP_LOSS_PCT)
        )

    def _buy(self, trader, price: float) -> None:
        trader.submit_order(
            shift.Order(shift.Order.Type.LIMIT_BUY, self.target, ORDER_SIZE, price)
        )
        self.has_position = True
        self.entry_price = price
        print(f"BUY {self.target} x{ORDER_SIZE} @ {price:.4f}")

    def _sell(self, trader, price: float) -> None:
        trader.cancel_all_pending_orders()
        trader.submit_order(
            shift.Order(shift.Order.Type.LIMIT_SELL, self.target, ORDER_SIZE, price)
        )
        self.has_position = False
        self.entry_price = None
        print(f"SELL {self.target} x{ORDER_SIZE} @ {price:.4f}")

    def trade_tick(self, trader) -> None:
        self.tick_count += 1

        book = self._best_price(trader)
        if book is None:
            return

        bid, ask, mid = book
        t_now = self.tick_count * INTERVAL_SEC

        p_pred, tau, source = self._predict_price(t_now)
        if p_pred is None:
            return

        obi = self._latest_obi()
        position = self._position(trader)

        buy_signal = p_pred > mid * (1 + PREDICTION_THRESHOLD)
        sell_signal = p_pred < mid * (1 - PREDICTION_THRESHOLD)
        obi_ok = obi > OBI_THRESHOLD
        stop_hit = self._stop_loss_hit(mid)

        if buy_signal and obi_ok:
            if not self.has_position and position < MAX_POSITION:
                self._buy(trader, bid)

        elif sell_signal or stop_hit:
            if self.has_position or position > 0:
                self._sell(trader, ask)

        if self.tick_count % 10 == 0:
            gap = p_pred - mid
            threshold_value = mid * PREDICTION_THRESHOLD
            print(
                f"[tick {self.tick_count}] {self.target} "
                f"mid={mid:.4f} pred={p_pred:.4f} gap={gap:+.4f} "
                f"threshold=±{threshold_value:.4f} source={source} tau={tau:.1f}s "
                f"obi={obi:+.3f} position={position}"
            )

    def stop(self) -> None:
        self.cache.stop()
        if self.collector is not None:
            self.collector.stop()


def main(trader) -> None:
    collector = BookCollector(trader)
    strategy = FFTPhaseLagStrategy(target=TARGET_TICKER)
    strategy.attach_collector(collector)

    collector.start()
    print("[main] Week 04 FFT phase-lag strategy started")

    try:
        while trader.is_connected():
            strategy.trade_tick(trader)
            time.sleep(INTERVAL_SEC)
    finally:
        strategy.stop()
        trader.cancel_all_pending_orders()
        print("[main] strategy stopped")


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
