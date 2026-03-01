import json
import os
from datetime import datetime, timezone
from pathlib import Path


class ModelOrchestrator:
    def __init__(self, storage_path: str = "data/model_state.json"):
        self.storage_path = Path(storage_path)
        self.state = {
            "champion_model": None,
            "champion_metrics": None,
            "history": [],
        }
        self._load_state()

        # Promotion gates: tune via env without code changes.
        self.min_trades = int(os.getenv("PROMOTION_MIN_TRADES", "10"))
        self.max_trades = int(os.getenv("PROMOTION_MAX_TRADES", "80"))
        self.min_win_rate = float(os.getenv("PROMOTION_MIN_WIN_RATE", "0.38"))
        self.min_total_return = float(os.getenv("PROMOTION_MIN_TOTAL_RETURN", "0.0"))
        self.max_drawdown_abs = float(os.getenv("PROMOTION_MAX_DRAWDOWN_ABS", "0.06"))
        self.min_sharpe = float(os.getenv("PROMOTION_MIN_SHARPE", "-0.25"))
        self.return_margin = float(os.getenv("PROMOTION_RETURN_MARGIN", "0.0015"))
        self.drawdown_margin = float(os.getenv("PROMOTION_DRAWDOWN_MARGIN", "0.003"))

    def _load_state(self):
        if self.storage_path.exists():
            try:
                with self.storage_path.open("r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.state.update(loaded)
            except Exception:
                pass

    def _save_state(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.storage_path.open("w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def _challenger_passes_hard_gates(self, challenger_metrics: dict) -> bool:
        trades = int(challenger_metrics.get("trades", 0) or 0)
        win_rate = float(challenger_metrics.get("win_rate", 0.0) or 0.0)
        total_return = float(challenger_metrics.get("total_return", -float("inf")) or -float("inf"))
        max_drawdown = float(challenger_metrics.get("max_drawdown", 0.0) or 0.0)
        sharpe_like = float(challenger_metrics.get("sharpe_like", -float("inf")) or -float("inf"))

        if trades < self.min_trades or trades > self.max_trades:
            return False
        if win_rate < self.min_win_rate:
            return False
        if total_return < self.min_total_return:
            return False
        if max_drawdown < -abs(self.max_drawdown_abs):
            return False
        if sharpe_like < self.min_sharpe:
            return False
        return True

    def should_promote(self, challenger_metrics: dict) -> bool:
        if not self._challenger_passes_hard_gates(challenger_metrics):
            return False

        champion = self.state.get("champion_metrics")
        if champion is None:
            return True

        challenger_return = float(challenger_metrics.get("total_return", -float("inf")) or -float("inf"))
        champion_return = float(champion.get("total_return", -float("inf")) or -float("inf"))
        challenger_drawdown = float(challenger_metrics.get("max_drawdown", -1.0) or -1.0)
        champion_drawdown = float(champion.get("max_drawdown", -1.0) or -1.0)
        challenger_sharpe = float(challenger_metrics.get("sharpe_like", -float("inf")) or -float("inf"))
        champion_sharpe = float(champion.get("sharpe_like", -float("inf")) or -float("inf"))

        materially_better_return = challenger_return >= (champion_return + self.return_margin)
        not_materially_worse_drawdown = challenger_drawdown >= (champion_drawdown - self.drawdown_margin)
        not_worse_sharpe = challenger_sharpe >= (champion_sharpe - 0.15)

        return materially_better_return and not_materially_worse_drawdown and not_worse_sharpe

    def promote(self, model_name: str, metrics: dict):
        self.state["champion_model"] = model_name
        self.state["champion_metrics"] = metrics

        history = self.state.get("history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "champion_model": model_name,
                "metrics": metrics,
            }
        )
        # Keep history bounded.
        self.state["history"] = history[-200:]
        self._save_state()

    def get_champion(self):
        return self.state.get("champion_model"), self.state.get("champion_metrics")
