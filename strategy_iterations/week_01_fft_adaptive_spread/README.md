# Week 01: FFT Adaptive-Spread Market Making Strategy

This folder documents a Week 01 strategy iteration from the Stevens High Frequency Trading Competition.

The strategy was an early market-making model that quoted AAPL and MSFT using top-of-book prices, order book imbalance, inventory controls, and an FFT-based adaptive spread filter.

## Strategy Idea

The model started from a simple passive market-making framework:

- collect best bid, best ask, and top-of-book size;
- estimate short-term mid-price variation from recent prices;
- use FFT energy to adjust the minimum acceptable spread;
- estimate a reservation price using mid-price, inventory, volatility, and order book imbalance;
- submit passive limit buy and sell orders when the spread was attractive enough;
- cancel stale quotes and flatten positions near the end of the session.

## Core Components

- Two-ticker strategy for AAPL and MSFT
- Rolling mid-price window
- FFT-based minimum spread adjustment
- Order book imbalance signal
- Inventory-aware reservation price
- Volatility-based quote width
- Maximum inventory limit
- End-of-session order cancellation and position flattening

## Notes

This is a cleaned and sanitized version for GitHub documentation. Private credentials and local configuration values are not included. The original weekly submission was part of a competition environment and required the SHIFT trading package and private connection files.
