from .state import (
    init_logger,
    current_logger,
    Logger
)


def init_super_openai(enable_caching=True, cache_size=1000):
    from .state import _internal_reset_global_state
    from .cache import init_cache
    init_cache(enable_caching, cache_size)
    _internal_reset_global_state()
