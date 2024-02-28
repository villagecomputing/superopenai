from .state import (
    init_logger,
    current_logger,
    _internal_reset_global_state,
    Logger
)

from .cache import init_cache


def init_super_openai(enable_caching=True, cache_size=1000):
    init_cache(enable_caching, cache_size)
    _internal_reset_global_state()
