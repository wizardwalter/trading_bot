from dataclasses import dataclass

@dataclass
class V3Config:
    core_floor_pct: float = 0.80
    max_scalp_tranche_pct: float = 0.20
    max_trims_per_day: int = 2
    friction_round_trip_pct: float = 0.0075
    support_resistance_lookback: int = 96  # 5m bars ~= 8h
    volume_spike_z: float = 2.0
