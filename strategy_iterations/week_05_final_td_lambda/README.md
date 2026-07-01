# TD(lambda) Reinforcement Learning Market-Making Strategy

This repository contains a TD(lambda)-based reinforcement learning market-making strategy developed for a simulated live trading competition environment.

## Overview

The strategy quotes bid and ask prices around the market mid-price while learning from inventory changes, fills, order-book conditions, and short-term price behavior. It uses tile coding for state representation and online TD(lambda) updates for action-value learning.

## Key Features

- TD(lambda) reinforcement learning core
- Tile coding for continuous state representation
- Inventory-aware bid/ask quoting
- Liquidity guards for thin and crossed books
- Pace guard to help satisfy minimum fill requirements
- Rolling retraining using recent market observations
- End-of-session position flattening

## State Variables

The model uses a five-dimensional state representation:

1. Normalized inventory
2. Order-book imbalance
3. Spread relative to recent volatility
4. Short-term vs. long-term volatility ratio
5. Ticks since last fill

## Action Space

Each action is a pair of bid and ask offsets from the mid-price. The model chooses from a discrete set of offset combinations and updates its action values online.

## Risk Controls

The strategy includes several non-RL safeguards:

- Skip quoting when the order book is too thin
- Widen quotes during crossed-book conditions
- Skew quotes to reduce inventory risk
- Enforce maximum inventory limits
- Cancel pending orders and flatten positions at the end of the session

## Requirements

Basic Python dependencies are listed in `requirements.txt`.

The live trading portion also requires access to the SHIFT trading package and private connection credentials. These credentials are not included in this repository and should be provided through environment variables.

## Environment Variables

Before running in a SHIFT-connected environment, set:

```bash
export SHIFT_USERNAME="your_username"
export SHIFT_CFG_FILE="initiator.cfg"
export SHIFT_PASSWORD="your_password"
```

Then run:

```bash
python td_lambda_market_maker.py
```

## Note

This project was built for a simulated trading competition environment. It is intended to demonstrate strategy design, reinforcement learning implementation, and market-making risk controls, not to provide production trading advice.
