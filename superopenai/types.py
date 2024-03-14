import time
import dataclasses
from typing import Dict, List, Union, Iterable, Optional, TypedDict
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
from superopenai.estimator import num_tokens_from_messages, get_cost

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
    cost: float
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
        table.field_names = ["Metric", "Value"]
        table.max_width = 80
        table.header = False
        table.horizontal_char = "-"
        table.add_row(["Number of Calls", self.num_calls])
        table.add_row(["Number Cached", self.num_cached], divider=True)
        table.add_row(["Cost", f"${self.cost}"], divider=True)
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
    cost: Optional[float] = None

    def __str__(self) -> str:
        result = []
        if self.cost is not None:
            result.append(f"- Cost: ${self.cost}")
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
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        model = params.get("model")
        return cls(
            input_messages=messages,
            input_args=params,
            output=response.choices,
            metadata=ChatCompletionLogMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=response.usage.total_tokens,
                start_time=start_time,
                latency=time.time() - start_time,
                cost=get_cost(prompt_tokens, completion_tokens, model)
            ),
            cached=cached
        )

    def __str__(self) -> str:
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.max_width = 80
        table.header = False
        table.align = "l"
        messages = []
        for msg in self.input_messages:
            if msg.get('content'):
                messages.append(f"- {msg['role']}: {msg['content']}")
            if msg.get('tool_calls'):
                for tool_call in msg.get('tool_calls'):
                    messages.append(
                        f"- {msg.get('role')}: ToolCall {tool_call.get('function')}")
            if msg.get('function_call'):
                messages.append(
                    f"- {msg.get('role')}: FunctionCall {msg.get('function_call')}")
        table.add_row(["Messages", "\n".join(messages)], divider=True)

        args = []
        for arg, value in self.input_args.items():
            args.append(f"- {arg}: {value}")
        table.add_row(["Arguments", "\n".join(args)], divider=True)

        outputs = []
        for choice in self.output:
            content = None
            if choice.message.content:
                content = f"- {choice.message.role}: {choice.message.content}"
            elif choice.message.tool_calls:
                content = f"- {choice.message.role}: {choice.message.tool_calls}"
            elif choice.message.function_call:
                content = f"- {choice.message.role}: {choice.message.function_call}"
            if content:
                outputs.append(content)
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
        output = [r.choices for r in all_responses]
        model = params.get("model")
        prompt_tokens = num_tokens_from_messages(messages, model)
        _, completion_tokens = StreamingChatCompletionLog._recover_outputs(
            output)
        return cls(
            input_messages=messages,
            input_args=params,
            output=output,
            metadata=ChatCompletionLogMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                start_time=start_time,
                latency=time.time() - start_time,
                cost=get_cost(prompt_tokens, completion_tokens, model)
            ),
            cached=cached
        )

    @staticmethod
    def _recover_outputs(output: List[List[ChunkChoice]]):
        # NOTE: completion tokens are probably being undercounted right now
        completion_tokens = 0
        max_index = max(
            choice.index for chunk in output for choice in chunk)
        recovered = [{} for _ in range(max_index + 1)]
        for chunk in output:
            for choice in chunk:
                idx = choice.index
                delta = choice.delta
                if delta.content:
                    completion_tokens += 1
                    recovered[idx]['content'] = recovered[idx].get(
                        'content', "") + delta.content
                if delta.role:
                    recovered[idx]['role'] = delta.role
                if delta.function_call:
                    recovered[idx]['function_call'] = delta.function_call
                if delta.tool_calls:
                    if not recovered[idx].get('tool_calls'):
                        recovered[idx]['tool_calls'] = [
                            {}] * len(delta.tool_calls)
                    for tool_delta in delta.tool_calls:
                        completion_tokens += 1
                        if tool_delta.function:
                            tool_idx = tool_delta.index
                            if tool_delta.function.name:
                                recovered[idx]['tool_calls'][tool_idx]['name'] = tool_delta.function.name
                            if tool_delta.function.arguments:
                                recovered[idx]['tool_calls'][tool_idx]['arguments'] = recovered[idx]['tool_calls'][tool_idx].get(
                                    "arguments", "") + tool_delta.function.arguments

        return recovered, completion_tokens

    def __str__(self) -> str:
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.max_width = 80
        table.header = False
        table.align = "l"
        msgs = []
        for msg in self.input_messages:
            if msg['content']:
                msgs.append(f"- {msg['role']}: {msg['content']}")
            if msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    msgs.append(
                        f"- {msg['role']}: ToolCall {tool_call['function']}")
            if msg['function_call']:
                msgs.append(
                    f"- {msg['role']}: FunctionCall {msg['function_call']}")
        table.add_row(["Messages", "\n".join(msgs)], divider=True)

        args = []
        for arg, value in self.input_args.items():
            args.append(f"- {arg}: {value}")
        table.add_row(["Arguments", "\n".join(args)], divider=True)

        recovered, _ = StreamingChatCompletionLog._recover_outputs(self.output)
        outputs = []
        for r in recovered:
            if r.get('content'):
                outputs.append(f"- {r.get('role')}: {r.get('content')}")
            if r.get('function_call'):
                outputs.append(
                    f"- {r.get('role')}: {r.get('function_call')}")
            if r.get('tool_calls'):
                for tool in r.get('tool_calls'):
                    outputs.append(
                        f"- {r.get('role')}: ToolCall {tool})")

        table.add_row(["Output", "\n".join(outputs)], divider=True)
        table.add_row(["Metadata", str(self.metadata)], divider=True)
        table.add_row(["Cached", str(self.cached)], divider=True)
        return table.get_string()
