import json
from pathlib import Path

class ModelOrchestrator:
    def __init__(self, storage_path: str = "data/model_state.json"):
        self.storage_path = Path(storage_path)
        self.state = {
            "champion_model": None,
            "champion_metrics": None,
        }
        self._load_state()

    def _load_state(self):
        if self.storage_path.exists():
            try:
                with self.storage_path.open("r") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save_state(self):
        try:
            with self.storage_path.open("w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def should_promote(self, challenger_metrics: dict) -> bool:
        # Basic promotion logic: challenger must have better test return and drawdown less negative
        champion = self.state.get("champion_metrics")
        if champion is None:
            return True

        challenger_return = challenger_metrics.get("total_return", -float("inf"))
        champion_return = champion.get("total_return", -float("inf"))
        challenger_drawdown = challenger_metrics.get("max_drawdown", 0)
        champion_drawdown = champion.get("max_drawdown", 0)

        return (
            challenger_return > champion_return and
            challenger_drawdown >= champion_drawdown
        )

    def promote(self, model_name: str, metrics: dict):
        self.state["champion_model"] = model_name
        self.state["champion_metrics"] = metrics
        self._save_state()

    def get_champion(self):
        return self.state.get("champion_model"), self.state.get("champion_metrics")
