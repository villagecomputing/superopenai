import time
from .cache import (
    get_cached,
    get_cached_streaming,
    get_cached_async,
    get_cached_streaming_async
)
from .types import ChatCompletionLog, StreamingChatCompletionLog


def wrap_create(create_fn, current_logger_fn):

    def create(self, *args, **kwargs):
        params = {**kwargs}
        messages = params.pop("messages", None)
        stream = kwargs.get("stream", False)
        logger = current_logger_fn()
        start_time = time.time()
        if stream:
            response, was_cached = get_cached_streaming(
                create_fn)(self, *args, **kwargs)

            def gen():
                all_results = []
                for item in response:
                    all_results.append(item)
                    yield item
                logger.log(StreamingChatCompletionLog.create(
                    messages=messages,
                    params=params,
                    all_responses=all_results,
                    cached=was_cached,
                    start_time=start_time
                ))
            return gen()
        else:
            response, was_cached = get_cached(
                create_fn)(self, *args, **kwargs)
            logger.log(ChatCompletionLog.create(
                messages=messages,
                params=params,
                response=response,
                cached=was_cached,
                start_time=start_time
            ))
            return response
    return create


async def wrap_acreate(acreate_fn, current_logger_fn):
    async def acreate(self, *args, **kwargs):
        params = {**kwargs}
        messages = params.pop("messages", None)
        stream = kwargs.get("stream", False)
        logger = current_logger_fn()
        start_time = time.time()
        if stream:
            response, was_cached = await get_cached_streaming_async(
                acreate_fn)(self, *args, **kwargs)

            async def gen():
                all_results = []
                async for item in response:
                    all_results.append(item)
                    yield item
                logger.log(StreamingChatCompletionLog.create(
                    messages=messages,
                    params=params,
                    all_responses=all_results,
                    cached=was_cached,
                    start_time=start_time
                ))
            return gen()
        else:
            response, was_cached = await get_cached_async(
                acreate_fn)(self, *args, **kwargs)
            logger.log(ChatCompletionLog.create(
                messages=messages,
                params=params,
                response=response,
                cached=was_cached,
                start_time=start_time
            ))
            return response
    return acreate
