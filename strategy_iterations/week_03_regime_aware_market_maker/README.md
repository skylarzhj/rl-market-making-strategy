# Week 03: Regime-Aware Market Making Strategy

This folder documents a Week 03 strategy iteration from the Stevens High Frequency Trading Competition.

The strategy explored a multi-ticker market-making framework that adjusted quoting behavior based on short-term volatility and market regime signals. It combined microprice-based fair value estimation, GARCH-style volatility tracking, and a regime classifier inspired by jump-model ideas.

## Strategy Idea

The strategy traded a basket of consumer/retail-related tickers and continuously quoted bid and ask prices around a microprice fair value estimate.

The main idea was to make the market maker more defensive under unfavorable market regimes:

- In a normal or bullish regime, the strategy used a higher inventory limit and tighter quotes.
- In a bearish or unstable regime, the strategy reduced inventory capacity and widened quotes.

## Core Components

- Multi-ticker market making
- Microprice-based fair value estimate
- Inventory-skewed reservation price
- GARCH(1,1)-style volatility update
- Online regime classification using downside deviation and Sortino-style features
- Dynamic programming with switching penalty for regime smoothing
- Different inventory and spread settings under bull and bear regimes
- End-of-session order cancellation and position-flattening logic

## Notes

This is a cleaned and sanitized version for GitHub documentation. Private credentials and local configuration values are not included. The original weekly submission was part of a competition environment and required the SHIFT trading package and private connection files.
