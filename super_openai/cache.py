import json
from cachetools import LRUCache

# TODO: support caching to disk and to a remote cache

# TODO: make these thread safe
_cache: LRUCache = None
_caching_enabled = False


def init_cache(enabled, size):
    global _cache
    global _caching_enabled
    _caching_enabled = enabled
    _cache = LRUCache(size)


# Note that the cache key uses the client_id to differentiate between different openai clients
# Requests with different client_ids will not share the same cache entry
def custom_key(self, *args, **kwargs):
    client_id = id(self._client)
    return json.dumps({**kwargs, **{"client_id": client_id}})


def get_cached(fn):
    if not _caching_enabled:
        return fn

    def cached_fn(self, *args, **kwargs):
        # Don't cache if temperature is non-zero because outputs are non-deterministic
        # Note: OpenAI sets temperature to 0 by default, but other providers may not
        temperature = kwargs.get("temperature", 0)
        if temperature != 0:
            return fn(self, *args, **kwargs), False
        key = custom_key(self, *args, **kwargs)
        if key in _cache:
            was_cached = True
            result = _cache[key]
        else:
            was_cached = False
            result = fn(self, *args, **kwargs)
            _cache[key] = result
        return result, was_cached

    return cached_fn


async def get_cached_async(fn):
    if not _caching_enabled:
        return fn

    async def cached_fn(self, *args, **kwargs):
        # Don't cache if temperature is non-zero because outputs are non-deterministic
        # Note: OpenAI sets temperature to 0 by default, but other providers may not
        temperature = kwargs.get("temperature", 0)
        if temperature != 0:
            return fn(self, *args, **kwargs), False
        key = custom_key(self, *args, **kwargs)
        if key in _cache:
            was_cached = True
            result = _cache[key]
        else:
            was_cached = False
            result = await fn(self, *args, **kwargs)
            _cache[key] = result
        return result, was_cached

    return cached_fn


def get_cached_streaming(fn):
    if not _caching_enabled:
        return fn

    def cached_fn(self, *args, **kwargs):
        key = custom_key(self, *args, **kwargs)
        if key in _cache:
            was_cached = True
            result = _cache[key]
        else:
            was_cached = False
            original_stream = fn(self, *args, **kwargs)

            def consume_stream():
                result_data = []
                try:
                    for data in original_stream:
                        result_data.append(data)
                        yield data
                finally:
                    _cache[key] = result_data
            result = consume_stream()

        return result, was_cached

    return cached_fn


async def get_cached_streaming_async(fn):
    if not _caching_enabled:
        return fn

    async def cached_fn(self, *args, **kwargs):
        key = custom_key(self, *args, **kwargs)
        if key in _cache:
            was_cached = True
            result = _cache[key]
        else:
            was_cached = False
            original_stream = fn(self, *args, **kwargs)

            async def consume_stream():
                result_data = []
                try:
                    async for data in original_stream:
                        result_data.append(data)
                        yield data
                finally:
                    _cache[key] = result_data
            result = consume_stream()

        return result, was_cached

    return cached_fn
