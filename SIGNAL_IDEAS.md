# Signal Ideas

## User ideas

- **DCA (Dollar Cost Average)** — basic (fixed interval buys) or dynamic (scale buy size by e.g. % below a moving average)
- **Heikin Ashi** — buy when HA candles flip green, sell when flip red
- **Mean reversion** — fade extreme extensions from a longer-term moving average
- **Seasonality** — e.g. "sell in May", month-of-year or day-of-week patterns
- **Buy the dip** — buy on % drawdown from recent high, sell on recovery

## Additional ideas

- **RSI overbought/oversold** — classic momentum oscillator; sell above 70, buy below 30. Configurable thresholds make it a good second signal to implement after MA crossover.
- **MACD crossover** — signal line crossover of the MACD histogram; a natural extension of the MA crossover already built.
- **Bollinger Band breakout/reversion** — price touching upper/lower bands can signal either a breakout (momentum) or reversion entry depending on the regime.
- **Volume spike** — unusual volume as a confirmation filter on top of a price signal; useful for detecting institutional activity.
- **Dual moving average with trend filter** — only take MA crossover signals in the direction of a longer-term trend (e.g. 200-day MA), reducing whipsaws in choppy markets.
- **Breakout from range** — buy new N-day highs, sell new N-day lows (Donchian channel / turtle trading style).
- **Multi-signal composite** — combine two or more signals with a voting/weighting mechanism; e.g. only buy if both RSI and MA crossover agree.
