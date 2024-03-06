_Logging and caching superpowers for your openai client_

## Introduction

super-openai was built to solve the following problems:

**Prompt and request visibility**. Many popular libraries like `langchain`, `guardrails`, `instructor` modify your prompts or even make additional requests under the hood. Sometimes this is useful, sometimes it's counter-productive. We believe it's good to adopt a "[show me the prompt](https://hamel.dev/blog/posts/prompt/)" attitude.

**Cost and latency tracking**. How much did my requests cost? How long did they take? These are important factors that affect the quality and feasibility of your software. Especially when you're chaining multiple LLM calls or building complex agent flows, it's important keep track of performance and understand which step is the bottleneck. This is useful both in development and production.

**Repeated identical requests**. During development, we often find ourselves changing one part of a pipeline and having to re-execute every LLM call on the entire dataset to see the new output. This unnecessarily slows down the development and iteration cycle. Caching ensures only the parts that change actually make new requests.

**Debugging complex chains/agents**. Complex chains or agents go wrong because of cascading failure. To debug the failure we need to inspect intermediate results and identify the source of error, then improve the prompt, try a different model, etc. It starts with logging and eyeballing the sequence of requests.

**Privacy, security and speed**. Many great tools exist to help you solve the above by sending your data to a remote server. But sometimes you need the data to stay local. Other times, you want to do some quick and dirty development without having to set up an api key, sign up for a service, understand their library and UI.

### Installation & basic usage

Run `pip install super-openai` or `poetry add super-openai`

To initialize super-openai, before initializing your openai client, run

```python
from super_openai import init_super_openai

init_super_openai()
```

This will monkey-patch the relevant functions in the `OpenAI` class. Then you can use `openai` library as usual with all the superpowers of super-openai

**Basic logging example**

```python
from openai import OpenAI
from super_openai import init_logger, init_super_openai

init_super_openai()
client = OpenAI()

with init_logger() as logger:
  client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "user", "content": "What's the capital of France?"}
    ])
  for log in logger.logs:
    print(log)
```

<details>
<summary> Expand to see output</summary>

```
+-----------+----------------------------------------------+
| Messages  | - user: What's the capital of France?        |
+-----------+----------------------------------------------+
| Arguments | - model: gpt-4-1106-preview                  |
+-----------+----------------------------------------------+
| Output    | - assistant: The capital of France is Paris. |
+-----------+----------------------------------------------+
| Metadata  | - Prompt tokens: 14                          |
|           | - Completion tokens: 7                       |
|           | - Total tokens: 21                           |
|           | - Start time: 1709679680.8683982             |
|           | - Latency: 0.9150938987731934                |
+-----------+----------------------------------------------+
| Cached    | False                                        |
+-----------+----------------------------------------------+
```

</details>

<br>
You can also avoid the context manager and directly manage starting and stopping loggers.

```python
logger = init_logger()
client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
  {"role": "user", "content": "What's the capital of France?"}
  ])
for log in logger.logs:
  print(log)
logger.end()
```

<details>
<summary> Expand to see output</summary>

```
+-----------+----------------------------------------------+
| Messages  | - user: What's the capital of France?        |
+-----------+----------------------------------------------+
| Arguments | - model: gpt-4-1106-preview                  |
+-----------+----------------------------------------------+
| Output    | - assistant: The capital of France is Paris. |
+-----------+----------------------------------------------+
| Metadata  | - Prompt tokens: 14                          |
|           | - Completion tokens: 7                       |
|           | - Total tokens: 21                           |
|           | - Start time: 1709679731.876042              |
|           | - Latency: 3.62396240234375e-05              |
+-----------+----------------------------------------------+
| Cached    | True                                         |
+-----------+----------------------------------------------+
```

</details>

<br>

Notice the second request's latency is almost 0 and `Cached` is `True`

## Logging

super-openai wraps the `OpenAI.chat.completions.create` and `AsyncOpenAI.chat.completions.create` functions and stores logs into a `super_openai.Logger` object. The following fields are captured and logged:

**Basic logging**

To start logging, call `init_logger()` either as a context manager `with init_logger() as logger` or as a simple function call. If not using a context manager, make sure to called `logger.end()`.

Every openai chat completion request will not be logged and logs will be stored in `logger.logs`. Each log is a `ChatCompletionLog` object containing the following fields:

- `input_messages`: a list of input prompts
- `input_args`: an object containing request arguments (model, streaming, temperature, etc.)
- `output`: a list of outputs (completion responses) produced by the LLM request
- `metadata`: metadata about the request
- `cached`: whether the response was returned from cache

**Token usage, cost and latency**

Inside the `metadata` field of each log you will find information about how many prompt and completions tokens were used, what the total cost was and the latency, ie. time between request being sent and response being received.

Cost is calculated based on prompt and completion token prices tokens defined in `estimator.py`. Only OpenAI models have pre-defined prices. If you're using non-OpenAI models, you can optionally specify a price dictionary when initializing `super-openai`. Prices are specified per 1M tokens in a tuple representing prompt and completion tokens respectively.

```python
from super_openai import init_super_openai

init_super_openai(cost_dict={
  'mistralai/Mixtral-8x7B-Instruct-v0.1': [0.5, 1.0]
})

```

**Streaming and async**

Logging works in streaming mode (setting `stream=True` in the chat completion request) as well as when using the async chat completion api.

In streaming mode, the output is a list of streamed chunks rather than a list of completion responses. All other fields are the same. The log object is a `StreamingChatCompletionLog` object.

**Function Calling and Tools**

`super-openai` works out of the box when using function calling or tools. The functions called and their arguments will be captured and printed in the `output` field. This works in streaming mode too.

**Statistics**

When you run a chain or agent with multiple LLM calls, it's useful to look at summary statistics over all the calls rather than individual ones.

To look at summary statistics, call `logger.summary_statistics()`

```python
with init_logger() as logger:
  client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "user", "content": "What's the capital of France?"}
    ]
  )
  print(logger.summary_statistics())
```

<details>
<summary> Expand to see output</summary>

```
+----------------------------+----------------------------+
|      Number of Calls       |             1              |
|       Number Cached        |             1              |
+----------------------------+----------------------------+
|       Prompt Tokens        |             14             |
|     Completion Tokens      |             7              |
|        Total Tokens        |             21             |
+----------------------------+----------------------------+
|   Prompt Tokens by Model   | {'gpt-4-1106-preview': 14} |
| Completion Tokens by Model | {'gpt-4-1106-preview': 7}  |
|   Total Tokens by Model    | {'gpt-4-1106-preview': 21} |
+----------------------------+----------------------------+
|       Total Latency        |    3.62396240234375e-05    |
|      Average Latency       |    3.62396240234375e-05    |
|  Average Latency (Cached)  |    3.62396240234375e-05    |
| Average Latency (Uncached) |             0              |
+----------------------------+----------------------------+
```

</details>

## Caching

`super-openai` caches all requests in-memory using `cachetools` and returns the cached response next time if all request parameters are exactly the same and the same `OpenAI` client is used.

Caching is automatically enabled when you called `init_super_openai` and applies both to regular `chat.completion.create` and async `chat.completion.create` requests. It works in both streaming and regular mode.

You can disable caching or change the cache size (default 1000) when initializating super-openai:

```python
from super_openai import init_super_openai

init_super_openai(enable_caching=True, cache_size=100)
```

## Using with langchain, etc.

super-openai is fully compatible with `langchain`, `llama-index`, `instructor`, `guidance`, `DSpy` and most other third party libraries.

This is particularly useful when you're doing local development with `langchain` and want to quickly inspect your chain runs, or understand what requests were made under the hood. For example:

```python
from super_openai import init_super_openai, init_logger
from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI

init_super_openai()

hard_question = "I have a 12 liter jug and a 6 liter jug.\
I want to measure 6 liters. How do I do it?"
prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo")

with init_logger() as logger:
  chain = SmartLLMChain(llm=llm, prompt=prompt,
                      n_ideas=2,
                      verbose=True)
  result = chain.run({})

print(logger.summary_statistics())
```

Output:

```
+----------------------------+-------------------------+
|      Number of Calls       |            4            |
|       Number Cached        |            0            |
+----------------------------+-------------------------+
|       Prompt Tokens        |           1183          |
|     Completion Tokens      |           591           |
|        Total Tokens        |           1774          |
+----------------------------+-------------------------+
|   Prompt Tokens by Model   | {'gpt-3.5-turbo': 1183} |
| Completion Tokens by Model |  {'gpt-3.5-turbo': 591} |
|   Total Tokens by Model    | {'gpt-3.5-turbo': 1774} |
+----------------------------+-------------------------+
|       Total Latency        |    8.826239109039307    |
|      Average Latency       |    2.2065597772598267   |
|  Average Latency (Cached)  |            0            |
| Average Latency (Uncached) |    2.2065597772598267   |
+----------------------------+-------------------------+
```

## Future work

- Port to TypeScript
- Simplifying retries
- Tracing
- Disk and remote caching
- Thread-safe caching
- Integrate with 3rd party hosted logging services

## Contributing

`super-openai` is free, open-source, and licensed under the MIT license. We welcome contributions from the community. You can always contribute by [giving us a star](https://github.com/villagecomputing/super-openai) :)

## License

`super-openai` is released under the MIT License. See the `LICENSE` file for more details.
