# from .rl import StrategyRL
from .strat_optuna import SearchStrategyOptuna
from .strat_proxy import SearchStrategyDaddyProxy
from .base import SearchStrategyBase


SEARCH_STRATEGY_MAP = {
    # "rl": StrategyRL,
    "optuna": SearchStrategyOptuna,
    "proxy": SearchStrategyDaddyProxy,
}


def get_search_strategy_cls(name: str) -> SearchStrategyBase:
    if name not in SEARCH_STRATEGY_MAP:
        raise ValueError(
            f"{name} must be defined in {list(SEARCH_STRATEGY_MAP.keys())}."
        )
    return SEARCH_STRATEGY_MAP[name]
