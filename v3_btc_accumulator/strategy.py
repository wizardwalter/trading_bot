from __future__ import annotations
from dataclasses import dataclass

@dataclass
class State:
    btc_units: float
    usd_units: float
    trims_today: int = 0

# Scaffold only: implement next commit with full signal engine.
def decide_action(*args, **kwargs):
    return {"action": "hold", "reason": "v3 scaffold"}
