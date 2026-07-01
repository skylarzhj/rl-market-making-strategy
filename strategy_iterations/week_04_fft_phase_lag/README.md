# Week 04 Submission

This folder contains the cleaned Week 04 strategy submission for the Stevens High Frequency Trading Competition.

The Week 04 idea was based on a high-frequency FFT phase-lag strategy. The strategy attempted to identify short-term leading/lagging relationships between CS1, CS2, and CS3, then use the leading ticker's phase information to predict CS2's short-term price movement.

## Strategy Idea

The strategy uses:

- 10 Hz order book data collection
- rolling price buffers
- Savitzky-Golay smoothing
- FFT decomposition of short-window price signals
- phase-lag estimation between tickers
- order book imbalance as a signal filter
- limit order execution with stop-loss and position limits

## Notes

This is a cleaned and sanitized version for GitHub documentation. Private credentials and local configuration values are not included. The original weekly submission was part of a competition environment and required the SHIFT trading package and private connection files.
