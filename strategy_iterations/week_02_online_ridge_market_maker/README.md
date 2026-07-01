# Week 02: Online Ridge Market Making Strategy

This folder documents a Week 02 strategy iteration from the Stevens High Frequency Trading Competition.

The strategy explored a short-horizon statistical market-making model for CSCO, MSFT, and NVDA. It used online feature collection, EWMA volatility tracking, ridge regression, inventory-aware quoting, and basic taker logic when the predicted edge was large enough.

## Strategy Idea

The model continuously estimated a short-term price signal and adjusted bid/ask quotes around a fair value estimate.

The main idea was:

- estimate microprice and order book imbalance from top-of-book data,
- track short-term returns and volatility with exponential moving averages,
- train an online ridge regression model on recent features,
- quote more aggressively when the predicted edge was favorable,
- reduce or close exposure near the end of the trading session.

## Core Components

- Multi-ticker strategy for CSCO, MSFT, and NVDA
- EWMA volatility model
- Online ridge regression
- Feature window for recent market observations
- Microprice and order book imbalance features
- Inventory-aware reservation price
- Dynamic inventory limits near session close
- Expected trade value filter
- Optional market-order taker logic for strong signals
- End-of-session cleanup and position flattening

## Notes

This is a cleaned and sanitized version for GitHub documentation. Private credentials and local configuration values are not included. The original weekly submission was part of a competition environment and required the SHIFT trading package and private connection files.
