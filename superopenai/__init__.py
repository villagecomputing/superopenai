from .state import (
    init_logger,
    current_logger,
    logged,
    Logger
)


def init_superopenai(
        enable_caching: bool = True,
        cache_size: int = 1000,
        cost_dict: dict = {}) -> None:
    """
    Initializes the superopenai library with optional caching and cost estimation settings.

    Args:
        enable_caching (bool): Flag to enable or disable caching of API responses. Defaults to True.
        cache_size (int): The maximum number of items to store in the cache. Only relevant if caching is enabled. Defaults to 1000.
        cost_dict (dict): A dictionary mapping model names to their respective cost per token. Used for cost estimation. Defaults to an empty dictionary.
    """
    from .state import _internal_reset_global_state
    from .cache import init_cache
    from .estimator import update_cost_dict
    init_cache(enable_caching, cache_size)
    update_cost_dict(cost_dict)
    _internal_reset_global_state()
