"""
TD(lambda) Reinforcement Learning Market-Making Strategy
======================================================

Single-file competition script for a 9-hour simulated live trading session.

Key differences from the earlier warmup/persistence version:
  - WARMUP_SEC = 0: the strategy begins quoting immediately at market open.
  - No pickle-based persistence: Q-values are kept in memory only.
  - If the process restarts, learned weights reset to zero.

Usage:
    python td_lambda_market_maker_clean.py

Workflow:
  0s       Cold start with theta initialized to zero and epsilon = 0.30.
  0 - 9h   Live market making with rolling retraining.
           - Every tick: quote with the RL policy and update online with TD(lambda).
           - Every 10 minutes: replay recent observations for rolling retraining.
           - Liquidity guards and pace guard are used as safety controls.
  9h       Cancel open orders, flatten positions, and exit.

Competition constraint:
  The strategy targets at least 200 total fills across two tickers during the
  9-hour session. The pace guard acts as a fallback if fill count falls behind.

Note:
  The live trading section requires the SHIFT trading package and private
  connection credentials. Credentials should be provided through environment
  variables and should never be committed to GitHub.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from datetime import datetime
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import numpy as np


# ============================================================
# Tile Coding
# ============================================================
# Fixed ticker-specific seeds keep the tile mapping stable across runs.
_TICKER_SEEDS = {"CS1": 1, "CS2": 2, "CS3": 3}


class TileCoder:
    """Independent per-dimension tile coder.

    The feature dimension is:
        n_dims * n_tilings * n_bins
    """

    def __init__(
        self,
        n_dims: int,
        n_tilings: int = 4,
        n_bins: int = 8,
        lows: Iterable[float] | None = None,
        highs: Iterable[float] | None = None,
        seed: int = 0,
    ):
        self.n_dims = n_dims
        self.n_tilings = n_tilings
        self.n_bins = n_bins
        self.lows = np.asarray(lows if lows is not None else [-1.0] * n_dims, dtype=float)
        self.highs = np.asarray(highs if highs is not None else [1.0] * n_dims, dtype=float)
        self.ranges = self.highs - self.lows + 1e-12

        rng = np.random.RandomState(seed)
        bin_widths = self.ranges / self.n_bins
        self.offsets = rng.uniform(0, 1, (n_tilings, n_dims)) * bin_widths
        self.feat_dim = n_dims * n_tilings * n_bins

    def encode(self, state: Iterable[float]) -> list[int]:
        """Return the active tile indices for a continuous state vector."""
        s = np.clip(np.asarray(state, dtype=float), self.lows, self.highs)
        active = []
        for tiling_idx in range(self.n_tilings):
            offset_s = s + self.offsets[tiling_idx]
            bin_idx = ((offset_s - self.lows) / self.ranges * self.n_bins).astype(int)
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            for dim_idx in range(self.n_dims):
                idx = (
                    dim_idx * (self.n_tilings * self.n_bins)
                    + tiling_idx * self.n_bins
                    + bin_idx[dim_idx]
                )
                active.append(idx)
        return active


# ============================================================
# Liquidity Guards
# ============================================================
def apply_liquidity_guards(
    my_bid: float,
    my_ask: float,
    mid: float,
    bid: float,
    ask: float,
    bid_vol: float,
    ask_vol: float,
    inventory: int,
    max_inv: int,
    tick: float = 0.01,
    min_book_size: int = 3,
) -> tuple[float | None, float | None]:
    """Apply non-RL safety rules before submitting quotes.

    Rules:
      1. If the book is too thin, do not quote.
      2. If the book is crossed, widen quotes to at least 20 ticks from mid.
      3. If inventory and order-book imbalance point in the same direction,
         skew quotes toward reducing inventory risk.
    """
    if bid_vol < min_book_size or ask_vol < min_book_size:
        return None, None

    if bid >= ask + 0.05:
        min_wide = 20 * tick
        if mid - my_bid < min_wide:
            my_bid = round(mid - min_wide, 2)
        if my_ask - mid < min_wide:
            my_ask = round(mid + min_wide, 2)

    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    inv_frac = inventory / max_inv if max_inv > 0 else 0.0

    if inv_frac > 0.5 and imbalance > 0.3:
        # Long inventory with buy-side pressure: shade quotes to sell down.
        my_ask = round(my_ask - 3 * tick, 2)
        my_bid = round(my_bid - 5 * tick, 2)
    elif inv_frac < -0.5 and imbalance < -0.3:
        # Short inventory with sell-side pressure: shade quotes to buy back.
        my_bid = round(my_bid + 3 * tick, 2)
        my_ask = round(my_ask + 5 * tick, 2)

    if my_bid >= my_ask:
        my_bid = round(mid - tick, 2)
        my_ask = round(mid + tick, 2)

    return my_bid, my_ask


# ============================================================
# TD(lambda) Market-Making Core
# ============================================================
class TDMarketMakerCore:
    """Pure TD(lambda) quoting logic with no dependency on SHIFT.

    Example:
        core = TDMarketMakerCore("CS1")
        my_bid, my_ask, action_idx, reward = core.step(
            mid, inventory, bid, ask, bid_vol, ask_vol
        )
    """

    def __init__(
        self,
        target_ticker: str = "CS1",
        alpha: float = 0.02,
        gamma: float = 0.99,
        lam: float = 0.9,
        eps: float = 0.30,
        eps_min: float = 0.05,
        eps_decay: float = 0.99995,
        eta: float = 0.3,
        max_inv: int = 20,
        tick_size: float = 0.01,
    ):
        self.target = target_ticker
        self.size = 1
        self.max_inv = max_inv
        self.tick_size = tick_size

        # Bid/ask offsets from mid, measured in ticks.
        # Offset 0 allows the pace guard to quote aggressively when needed.
        self.action_offsets = [0, 1, 2, 5, 10, 20, 40]
        self.actions = [(b, a) for b in self.action_offsets for a in self.action_offsets]
        self.n_actions = len(self.actions)

        seed = _TICKER_SEEDS.get(target_ticker, hash(target_ticker) % 1000)
        self.tc = TileCoder(
            n_dims=5,
            n_tilings=4,
            n_bins=8,
            lows=[-1] * 5,
            highs=[1] * 5,
            seed=seed,
        )

        self.theta = np.zeros((self.n_actions, self.tc.feat_dim))
        self.e = np.zeros_like(self.theta)

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eta = eta

        self.mids = deque(maxlen=200)
        self.prev_active: list[int] | None = None
        self.prev_action_idx: int | None = None
        self.prev_inventory = 0
        self.prev_mid: float | None = None
        self.prev_equity = 0.0
        self.cash = 0.0
        self.last_my_bid: float | None = None
        self.last_my_ask: float | None = None
        self.ticks_since_fill = 0
        self.step_count = 0

    def _compute_state(
        self,
        mid: float,
        inventory: int,
        bid: float,
        ask: float,
        bid_vol: float,
        ask_vol: float,
    ) -> np.ndarray:
        inv_norm = np.clip(inventory / self.max_inv, -1, 1)

        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
        imbalance = np.clip(imbalance, -1, 1)

        spread = max(ask - bid, 0.0)
        long_vol = np.std(self.mids) if len(self.mids) >= 20 else max(spread, 0.01)
        spread_norm = np.clip(spread / (long_vol + 1e-6) - 1.0, -1, 1)

        if len(self.mids) >= 20:
            short_vol = np.std(list(self.mids)[-20:])
            vol_ratio = np.clip(short_vol / (long_vol + 1e-6) - 1.0, -1, 1)
        else:
            vol_ratio = 0.0

        tsf_norm = np.clip(self.ticks_since_fill / 30.0 - 1.0, -1, 1)
        return np.array([inv_norm, imbalance, spread_norm, vol_ratio, tsf_norm])

    def _q_values(self, active: list[int]) -> np.ndarray:
        return self.theta[:, active].sum(axis=1)

    def _choose_action(self, active: list[int]) -> int:
        if np.random.random() < self.eps:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self._q_values(active)))

    def _step_reward(self, mid: float, inventory: int) -> float:
        d_inv = inventory - self.prev_inventory

        if d_inv > 0 and self.last_my_bid is not None:
            self.cash -= d_inv * self.last_my_bid
            self.ticks_since_fill = 0
        elif d_inv < 0 and self.last_my_ask is not None:
            self.cash += (-d_inv) * self.last_my_ask
            self.ticks_since_fill = 0
        else:
            self.ticks_since_fill += 1

        equity = self.cash + inventory * mid
        if self.prev_mid is None:
            self.prev_equity = equity
            return 0.0

        d_equity = equity - self.prev_equity
        d_mid = mid - self.prev_mid
        speculative_profit = self.prev_inventory * d_mid
        reward = d_equity - self.eta * max(0.0, speculative_profit)
        self.prev_equity = equity
        return reward

    def _td_update(self, reward: float, next_active: list[int], next_action_idx: int) -> None:
        if self.prev_active is None or self.prev_action_idx is None:
            return

        q_sa = self.theta[self.prev_action_idx, self.prev_active].sum()
        q_next = self.theta[next_action_idx, next_active].sum()
        delta = reward + self.gamma * q_next - q_sa

        self.e *= self.gamma * self.lam
        self.e[self.prev_action_idx, self.prev_active] = 1.0
        self.theta += (self.alpha / self.tc.n_tilings) * delta * self.e

    def step(
        self,
        mid: float,
        inventory: int,
        bid: float,
        ask: float,
        bid_vol: float,
        ask_vol: float,
    ) -> tuple[float, float, int, float]:
        """Process one market tick and return the next quote."""
        self.mids.append(mid)
        reward = self._step_reward(mid, inventory)

        state = self._compute_state(mid, inventory, bid, ask, bid_vol, ask_vol)
        active = self.tc.encode(state)
        action_idx = self._choose_action(active)
        self._td_update(reward, active, action_idx)

        bid_off, ask_off = self.actions[action_idx]
        my_bid = round(mid - bid_off * self.tick_size, 2)
        my_ask = round(mid + ask_off * self.tick_size, 2)

        self.last_my_bid = my_bid
        self.last_my_ask = my_ask
        self.prev_active = active
        self.prev_action_idx = action_idx
        self.prev_inventory = inventory
        self.prev_mid = mid
        self.step_count += 1

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return my_bid, my_ask, action_idx, reward


# ============================================================
# Optional SHIFT Import
# ============================================================
# Delayed import allows the core model to be imported in a local environment
# where the SHIFT package is not installed.
try:
    import shift

    HAS_SHIFT = True
except ImportError:
    HAS_SHIFT = False


# ============================================================
# Session Parameters
# ============================================================
WARMUP_SEC = 0
TOTAL_SESSION_SEC = 9 * 3600
TARGET_FILLS_TOTAL = 200
TARGET_FILLS_PER_TICKER = 110
PACE_GUARD_BEHIND_FRAC = 0.7

WARMUP_RETRAIN_EPOCHS = 10
ROLLING_RETRAIN_EPOCHS = 3
ROLLING_RETRAIN_INTERVAL_SEC = 600
MAX_OBSERVATION_BUFFER = 4000

TICK_SLEEP_SEC = 0.5

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class LiveMarketMaker:
    """Live trading wrapper for a single ticker."""

    def __init__(self, target_ticker: str):
        self.target = target_ticker
        self.core = TDMarketMakerCore(
            target_ticker=target_ticker,
            eps=0.30,
            eps_min=0.05,
            eps_decay=0.99995,
            eta=0.3,
            max_inv=20,
        )

        self.start_time: float | None = None
        self.warmup_ended = False
        self.observations: list[dict[str, float]] = []
        self.fill_count = 0
        self.prev_inventory = 0
        self.next_rolling_retrain: float | None = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.log_path = os.path.join(LOG_DIR, f"{target_ticker}_{timestamp}.csv")
        self._init_log()

    def _init_log(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t",
                    "mid",
                    "bid",
                    "ask",
                    "bid_vol",
                    "ask_vol",
                    "inventory",
                    "my_bid",
                    "my_ask",
                    "fill_count",
                    "phase",
                ]
            )

    def _log(self, row: list[object]) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def _pace_guard_active(self) -> bool:
        live_sec = self._elapsed() - WARMUP_SEC
        if live_sec <= 600:
            return False

        live_total = TOTAL_SESSION_SEC - WARMUP_SEC
        expected_fills = TARGET_FILLS_PER_TICKER * live_sec / live_total
        return self.fill_count < expected_fills * PACE_GUARD_BEHIND_FRAC

    def _retrain(self, epochs: int, label: str = "retrain") -> None:
        """Replay recent observations to update the Q table.

        A simple mid-cross rule is used to create synthetic fill events.
        """
        rows = self.observations
        if len(rows) < 100:
            print(f"[{self.target}] {label} skipped (only {len(rows)} rows)")
            return

        if len(rows) > MAX_OBSERVATION_BUFFER:
            rows = rows[-MAX_OBSERVATION_BUFFER:]

        start = time.time()
        saved_eps = self.core.eps
        self.core.eps = min(0.3, max(saved_eps, 0.1))
        total_reward = 0.0
        total_fills = 0

        for _ in range(epochs):
            inventory = 0
            self.core.cash = 0.0
            self.core.prev_equity = 0.0
            self.core.prev_mid = None
            self.core.prev_active = None
            self.core.prev_action_idx = None
            self.core.e[:] = 0

            episode_reward = 0.0
            episode_fills = 0
            for i in range(len(rows) - 1):
                row = rows[i]
                next_row = rows[i + 1]
                my_bid, my_ask, _, reward = self.core.step(
                    row["mid"],
                    inventory,
                    row["bid"],
                    row["ask"],
                    row["bid_vol"],
                    row["ask_vol"],
                )
                episode_reward += reward

                if my_bid > 0 and next_row["mid"] <= my_bid and inventory < self.core.max_inv:
                    inventory += 1
                    episode_fills += 1
                if my_ask > 0 and next_row["mid"] >= my_ask and inventory > -self.core.max_inv:
                    inventory -= 1
                    episode_fills += 1

            total_reward += episode_reward
            total_fills += episode_fills

        self.core.e[:] = 0
        self.core.prev_active = None
        self.core.prev_action_idx = None
        self.core.eps = saved_eps

        elapsed = time.time() - start
        print(
            f"[{self.target}] {label} done: rows={len(rows)} ep={epochs} "
            f"sim_fills={total_fills} reward={total_reward:+.1f} "
            f"took={elapsed:.1f}s epsilon={self.core.eps:.3f}"
        )

    def tick(self, trader) -> None:
        if self.start_time is None:
            self.start_time = time.time()

        best_price = trader.get_best_price(self.target)
        bid = best_price.get_bid_price()
        ask = best_price.get_ask_price()
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2
        try:
            bid_vol = float(best_price.get_bid_size())
            ask_vol = float(best_price.get_ask_size())
        except Exception:
            bid_vol, ask_vol = 5.0, 5.0

        elapsed = self._elapsed()

        self.observations.append(
            {
                "mid": mid,
                "bid": bid,
                "ask": ask,
                "bid_vol": max(bid_vol, 1.0),
                "ask_vol": max(ask_vol, 1.0),
            }
        )
        if len(self.observations) > MAX_OBSERVATION_BUFFER * 2:
            self.observations = self.observations[-MAX_OBSERVATION_BUFFER:]

        if elapsed < WARMUP_SEC:
            self.core.mids.append(mid)
            self._log([int(elapsed), mid, bid, ask, bid_vol, ask_vol, 0, "", "", 0, "warmup"])
            return

        if not self.warmup_ended:
            self.warmup_ended = True
            self._retrain(epochs=WARMUP_RETRAIN_EPOCHS, label="cold-start-retrain")
            self.next_rolling_retrain = elapsed + ROLLING_RETRAIN_INTERVAL_SEC

        if self.next_rolling_retrain is not None and elapsed >= self.next_rolling_retrain:
            self._retrain(epochs=ROLLING_RETRAIN_EPOCHS, label=f"rolling-retrain@{int(elapsed / 60)}min")
            self.next_rolling_retrain = elapsed + ROLLING_RETRAIN_INTERVAL_SEC

        inventory = trader.get_portfolio_item(self.target).get_shares()
        d_inv = abs(inventory - self.prev_inventory)
        if d_inv > 0:
            self.fill_count += d_inv

        my_bid, my_ask, action_idx, reward = self.core.step(
            mid, inventory, bid, ask, bid_vol, ask_vol
        )

        pace_override = self._pace_guard_active()
        if pace_override:
            if np.random.random() < 0.5:
                my_bid = round(mid - 0.01, 2)
                my_ask = round(mid + np.random.randint(1, 4) * 0.01, 2)
            else:
                my_bid = round(mid - np.random.randint(1, 4) * 0.01, 2)
                my_ask = round(mid + 0.01, 2)

        guarded_bid, guarded_ask = apply_liquidity_guards(
            my_bid,
            my_ask,
            mid,
            bid,
            ask,
            bid_vol,
            ask_vol,
            inventory,
            self.core.max_inv,
            tick=self.core.tick_size,
        )

        trader.cancel_all_pending_orders()
        if guarded_bid is not None and guarded_ask is not None:
            if inventory < self.core.max_inv and guarded_bid > 0:
                trader.submit_order(
                    shift.Order(shift.Order.Type.LIMIT_BUY, self.target, self.core.size, guarded_bid)
                )
            if inventory > -self.core.max_inv and guarded_ask > 0:
                trader.submit_order(
                    shift.Order(shift.Order.Type.LIMIT_SELL, self.target, self.core.size, guarded_ask)
                )

        self.prev_inventory = inventory

        phase = "live_pace" if pace_override else "live"
        if guarded_bid is None:
            phase = "skip_thin"

        self._log(
            [
                int(elapsed),
                mid,
                bid,
                ask,
                bid_vol,
                ask_vol,
                inventory,
                guarded_bid or "",
                guarded_ask or "",
                self.fill_count,
                phase,
            ]
        )

        if self.core.step_count % 20 == 0:
            bid_offset, ask_offset = self.core.actions[action_idx]
            flags = ""
            if pace_override:
                flags += "P"
            if guarded_bid is None:
                flags += "T"
            print(
                f"[{self.target}] t={int(elapsed)}s mid={mid:.2f} "
                f"inv={inventory:+d} a=({bid_offset:>2d},{ask_offset:>2d}) "
                f"quote=[{guarded_bid},{guarded_ask}] r={reward:+.2f} "
                f"epsilon={self.core.eps:.3f} "
                f"fills={self.fill_count}/{TARGET_FILLS_PER_TICKER} [{flags}]"
            )


def main(trader) -> None:
    print(f"=== TD(lambda) market maker started at {datetime.now()} ===")
    print(f"    WARMUP {WARMUP_SEC}s -> LIVE ({TOTAL_SESSION_SEC - WARMUP_SEC}s)")
    print(
        f"    Target: at least {TARGET_FILLS_TOTAL} fills total, "
        f"{TARGET_FILLS_PER_TICKER}/ticker"
    )

    strategies = {
        "CS1": LiveMarketMaker("CS1"),
        "CS2": LiveMarketMaker("CS2"),
    }
    last_status = time.time()
    session_start = time.time()

    try:
        while trader.is_connected():
            if time.time() - session_start > TOTAL_SESSION_SEC + 60:
                print("=== session time limit reached; flattening positions ===")
                break

            for ticker, strategy in strategies.items():
                try:
                    strategy.tick(trader)
                except Exception as exc:
                    print(f"[{ticker}] tick error: {exc}")

            if time.time() - last_status > 60:
                last_status = time.time()
                total_fills = sum(strategy.fill_count for strategy in strategies.values())
                print(f"--- total_fills={total_fills}/{TARGET_FILLS_TOTAL} ---")

            time.sleep(TICK_SLEEP_SEC)
    finally:
        try:
            trader.cancel_all_pending_orders()
            for ticker in strategies:
                pos = trader.get_portfolio_item(ticker).get_shares()
                if pos != 0:
                    order_type = (
                        shift.Order.Type.MARKET_SELL if pos > 0 else shift.Order.Type.MARKET_BUY
                    )
                    trader.submit_order(shift.Order(order_type, ticker, abs(pos)))
                    print(f"[{ticker}] final flatten: {pos} -> 0")
        except Exception as exc:
            print(f"flatten error: {exc}")

        total = sum(strategy.fill_count for strategy in strategies.values())
        print(f"=== FINAL: total fills = {total} (target {TARGET_FILLS_TOTAL}) ===")


if __name__ == "__main__":
    if not HAS_SHIFT:
        raise ImportError("shift package not found; this script must run in a SHIFT-connected environment")

    username = os.getenv("SHIFT_USERNAME", "four-sigma")
    cfg_file = os.getenv("SHIFT_CFG_FILE", "initiator.cfg")
    password = os.getenv("SHIFT_PASSWORD")

    if not password:
        raise RuntimeError("SHIFT_PASSWORD environment variable is required")

    with shift.Trader(username) as trader:
        try:
            trader.connect(cfg_file, password)
            trader.sub_all_order_book()
            main(trader)
        except Exception as exc:
            print(f"Runtime error: {exc}")
