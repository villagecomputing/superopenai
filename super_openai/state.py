import contextvars
import asyncio
from typing import List, Union
from openai.resources.chat.completions import Completions, AsyncCompletions
from .types import *
from .wrap_openai import wrap_create, wrap_acreate


class Logger:
    def __init__(self, set_current: bool = True):
        self.logs: List[Union[ChatCompletionLog,
                              StreamingChatCompletionLog]] = []
        if set_current:
            self.start()

    def log(self, log: Union[ChatCompletionLog, StreamingChatCompletionLog]):
        self.logs.append(log)

    def summary_statistics(self) -> SummaryStatistics:
        num_calls = len(self.logs)
        num_cached = sum(log.cached for log in self.logs)
        cost = sum(
            log.metadata.cost for log in self.logs if log.metadata.cost is not None)
        prompt_tokens = sum(
            log.metadata.prompt_tokens for log in self.logs if log.metadata.prompt_tokens is not None)
        completion_tokens = sum(
            log.metadata.completion_tokens for log in self.logs if log.metadata.completion_tokens is not None)
        total_tokens = sum(
            log.metadata.total_tokens for log in self.logs if log.metadata.total_tokens is not None)
        prompt_tokens_by_model = {}
        completion_tokens_by_model = {}
        total_tokens_by_model = {}
        total_latency = sum(log.metadata.latency for log in self.logs)
        avg_latency = total_latency / num_calls if num_calls > 0 else 0
        avg_latency_cached = sum(
            log.metadata.latency for log in self.logs if log.cached) / num_cached if num_cached > 0 else 0
        avg_latency_uncached = sum(log.metadata.latency for log in self.logs if not log.cached) / (
            num_calls - num_cached) if num_calls - num_cached > 0 else 0

        for log in self.logs:
            model = log.input_args.get("model")
            if model not in prompt_tokens_by_model:
                prompt_tokens_by_model[model] = 0
                completion_tokens_by_model[model] = 0
                total_tokens_by_model[model] = 0
            if log.metadata.prompt_tokens is not None:
                prompt_tokens_by_model[model] += log.metadata.prompt_tokens
            if log.metadata.completion_tokens is not None:
                completion_tokens_by_model[model] += log.metadata.completion_tokens
            if log.metadata.total_tokens is not None:
                total_tokens_by_model[model] += log.metadata.total_tokens

        return SummaryStatistics(
            num_calls=num_calls,
            num_cached=num_cached,
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_by_model=prompt_tokens_by_model,
            completion_tokens_by_model=completion_tokens_by_model,
            total_tokens_by_model=total_tokens_by_model,
            total_latency=total_latency,
            avg_latency=avg_latency,
            avg_latency_cached=avg_latency_cached,
            avg_latency_uncached=avg_latency_uncached
        )

    def start(self):
        if current_logger() != self:
            self._context_token = _state.current_logger.set(self)

    def end(self):
        if current_logger() == self:
            _state.current_logger.reset(self._context_token)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, callback):
        self.end()
        del type, value, callback


class NoopLogger(Logger):
    def log(self, data):
        pass

    def start(self):
        pass

    def end(self):
        pass


NOOP_LOGGER = NoopLogger()


class SuperOpenAIState:
    def __init__(self):
        self.current_logger = contextvars.ContextVar(
            "super_openai_current_logger", default=NOOP_LOGGER)
        self.wrapped = False
        self.create_fn = None
        self.acreate_fn = None
        self._wrap()

    def _wrap(self):
        if self.wrapped:
            # Prevent double-wrapping
            return
        create_fn = Completions.create
        acreate_fn = AsyncCompletions.create
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn
        Completions.create = wrap_create(create_fn, current_logger)
        AsyncCompletions.create = wrap_acreate(acreate_fn, current_logger)

    def reset(self):
        self.current_logger = contextvars.ContextVar(
            "super_openai_current_logger", default=NOOP_LOGGER)


def init_logger():
    return Logger()


def current_logger() -> Logger:
    return _state.current_logger.get()


def logged(fn):
    if asyncio.iscoroutinefunction(fn):
        async def async_wrapper(*args, **kwargs):
            async with init_logger() as logger:
                return await fn(*args, **kwargs), logger
        return async_wrapper
    else:
        def wrapper(*args, **kwargs):
            with init_logger() as logger:
                return fn(*args, **kwargs), logger
        return wrapper


_state: Optional[SuperOpenAIState] = None


def _internal_reset_global_state():
    global _state
    if _state is None:
        # The state should only be initalized once in the life of the application
        _state = SuperOpenAIState()
    else:
        _state.reset()
