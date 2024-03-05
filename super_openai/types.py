import time
import dataclasses
from typing import Dict, List, Union, Iterable, Optional, overload, TypedDict
from typing_extensions import Literal
from openai._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
import httpx
from prettytable import PrettyTable

ModelType = Union[
    str,
    Literal[
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613",
    ],
],


class ChatCompletionCreateOptionalArgs(TypedDict, total=False):
    frequency_penalty: Optional[float] | NotGiven
    function_call: completion_create_params.FunctionCall | NotGiven
    functions: Iterable[completion_create_params.Function] | NotGiven
    logit_bias: Optional[Dict[str, int]] | NotGiven
    logprobs: Optional[bool] | NotGiven
    max_tokens: Optional[int] | NotGiven
    n: Optional[int] | NotGiven
    presence_penalty: Optional[float] | NotGiven
    response_format: completion_create_params.ResponseFormat | NotGiven
    seed: Optional[int] | NotGiven
    stop: Union[Optional[str], List[str]] | NotGiven
    stream: Optional[Literal[False]] | NotGiven
    temperature: Optional[float] | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    tools: Iterable[ChatCompletionToolParam] | NotGiven
    top_logprobs: Optional[int] | NotGiven
    top_p: Optional[float] | NotGiven
    user: str | NotGiven
    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | httpx.Timeout | None | NotGiven


@dataclasses.dataclass
class SummaryStatistics:
    num_calls: int
    num_cached: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_by_model: Dict[str, int]
    completion_tokens_by_model: Dict[str, int]
    total_tokens_by_model: Dict[str, int]
    total_latency: float
    avg_latency: float
    avg_latency_cached: float
    avg_latency_uncached: float

    def __str__(self) -> str:
        table = PrettyTable()
        table.header = False
        table.horizontal_char = "-"
        table.add_row(["Number of Calls", self.num_calls])
        table.add_row(["Number Cached", self.num_cached], divider=True)
        table.add_row(["Prompt Tokens", self.prompt_tokens])
        table.add_row(["Completion Tokens", self.completion_tokens])
        table.add_row(["Total Tokens", self.total_tokens], divider=True)
        table.add_row(["Prompt Tokens by Model", self.prompt_tokens_by_model])
        table.add_row(["Completion Tokens by Model",
                      self.completion_tokens_by_model])
        table.add_row(["Total Tokens by Model",
                      self.total_tokens_by_model], divider=True)
        table.add_row(["Total Latency", self.total_latency])
        table.add_row(["Average Latency", self.avg_latency])
        table.add_row(["Average Latency (Cached)", self.avg_latency_cached])
        table.add_row(["Average Latency (Uncached)",
                      self.avg_latency_uncached])
        return table.get_string()


@dataclasses.dataclass
class ChatCompletionLogMetadata:
    start_time: float
    latency: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def __str__(self) -> str:
        result = []
        if self.prompt_tokens is not None:
            result.append(f"- Prompt tokens: {self.prompt_tokens}")
        if self.completion_tokens is not None:
            result.append(f"- Completion tokens: {self.completion_tokens}")
        if self.total_tokens is not None:
            result.append(f"- Total tokens: {self.total_tokens}")
        result.append(f"- Start time: {self.start_time}")
        result.append(f"- Latency: {self.latency}")
        return "\n".join(result)


@dataclasses.dataclass
class ChatCompletionLog:
    input_messages: Iterable[ChatCompletionMessageParam]
    input_args: ChatCompletionCreateOptionalArgs
    output: List[Choice]
    metadata: ChatCompletionLogMetadata
    cached: bool

    @classmethod
    def create(cls,
               messages: Iterable[ChatCompletionMessageParam],
               params: ChatCompletionCreateOptionalArgs,
               response: ChatCompletion,
               cached: bool,
               start_time: float
               ):
        return cls(
            input_messages=messages,
            input_args=params,
            # for now, we don't handle tools/function calls
            output=response.choices,
            metadata=ChatCompletionLogMetadata(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                start_time=start_time,
                latency=time.time() - start_time
            ),
            cached=cached
        )

    def __str__(self) -> str:
        table = PrettyTable()
        table.header = False
        table.align = "l"
        messages = []
        for msg in self.input_messages:
            messages.append(f"- {msg['role']}: {msg['content']}")
        table.add_row(["Messages", "\n".join(messages)], divider=True)

        args = []
        for arg, value in self.input_args.items():
            args.append(f"- {arg}: {value}")
        table.add_row(["Arguments", "\n".join(args)], divider=True)

        outputs = []
        for choice in self.output:
            outputs.append(
                f"- {choice.message.role}: {choice.message.content}")
        table.add_row(["Output", "\n".join(outputs)], divider=True)
        table.add_row(["Metadata", str(self.metadata)], divider=True)
        table.add_row(["Cached", str(self.cached)])

        return table.get_string()


@dataclasses.dataclass
class StreamingChatCompletionLog():
    input_messages: Iterable[ChatCompletionMessageParam]
    input_args: ChatCompletionCreateOptionalArgs
    output: List[List[ChunkChoice]]
    metadata: ChatCompletionLogMetadata
    cached: bool

    @classmethod
    def create(cls,
               messages: Iterable[ChatCompletionMessageParam],
               params: ChatCompletionCreateOptionalArgs,
               all_responses: List[ChatCompletionChunk],
               cached: bool,
               start_time: float
               ):
        return cls(
            input_messages=messages,
            input_args=params,
            output=[r.choices for r in all_responses],
            metadata=ChatCompletionLogMetadata(
                # The openai api doesn't return token counts for streaming
                # TODO: implement manual token count calculation for streaming
                start_time=start_time,
                latency=time.time() - start_time
            ),
            cached=cached
        )

    def __str__(self) -> str:
        table = PrettyTable()
        table.header = False
        table.align = "l"
        msgs = []
        for msg in self.input_messages:
            msgs.append(f"- {msg['role']}: {msg['content']}")
        table.add_row(["Messages", "\n".join(msgs)], divider=True)

        args = []
        for arg, value in self.input_args.items():
            args.append(f"- {arg}: {value}")
        table.add_row(["Arguments", "\n".join(args)], divider=True)

        outputs = []
        num_responses = len(self.output[0])
        for index in range(num_responses):
            concatenated_content = "".join(
                [chunk_list[index].delta.content or "" if
                 index < len(chunk_list) else ""
                 for chunk_list in self.output])
            outputs.append(f"- {concatenated_content}")
        table.add_row(["Output", "\n".join(outputs)], divider=True)
        table.add_row(["Metadata", str(self.metadata)], divider=True)
        table.add_row(["Cached", str(self.cached)], divider=True)
        return table.get_string()
