# RL Market-Making Strategy Archive

This repository documents five strategy iterations from the Stevens High Frequency Trading Competition.

The project evolved from simple rule-based market making to online statistical models, regime-aware quoting, FFT phase-lag signals, and finally a TD(λ)-based reinforcement learning market-making strategy.

## Strategy Iterations

| Folder | Strategy Idea |
|---|---|
| `strategy_iterations/week_01_fft_adaptive_spread/` | Early FFT adaptive-spread market maker using rolling mid-price signals, order book imbalance, and inventory controls |
| `strategy_iterations/week_02_online_ridge_market_maker/` | Online ridge regression market maker using microprice, imbalance, EWMA volatility, and short-term return features |
| `strategy_iterations/week_03_regime_aware_market_maker/` | Regime-aware market maker using microprice, GARCH-style volatility, downside-risk features, and bull/bear quote adjustment |
| `strategy_iterations/week_04_fft_phase_lag/` | High-frequency FFT phase-lag strategy using lead-lag relationships between CS1, CS2, and CS3 with order book imbalance filtering |
| `strategy_iterations/week_05_final_td_lambda/` | Final TD(λ) reinforcement learning market-making strategy with tile coding, liquidity guards, rolling retraining, and pace control |

## Final Strategy

The final version is located in:

`strategy_iterations/week_05_final_td_lambda/`

It uses a TD(λ)-based reinforcement learning framework to quote bid and ask prices around the market mid-price. The strategy includes tile coding, online SARSA-style updates, inventory-aware quoting, liquidity guards, rolling retraining, and end-of-session position flattening.

## Notes

This project was developed for a simulated trading competition environment. The live trading components require the SHIFT trading package and private configuration files, which are not included in this repository.

Private credentials, account information, and local configuration values have been removed.
