# V3 BTC Accumulator (Core + Scalp Overlay)

Objective: maximize BTC units over time (not USD PnL).

## Principles
- Keep core BTC position (never fully sell)
- Trade only scalp tranche around support/resistance
- Regime-adaptive behavior:
  - normal: small trims/rebuys
  - expansion: trim less, ride trend
  - breakdown: trim more, rebuy on stabilization
- Score by net BTC accumulation after friction

## Initial defaults
- core_floor_pct: 0.80
- max_scalp_tranche_pct: 0.20
- max_trims_per_day: 2
- friction_round_trip_pct: 0.75%
