_Logging and caching superpowers for your openai client_

## Introduction

It aims to solve the following problems:

- Prompt and request visibility
- Token usage and latency counting across multiple requests
- Caching repeated identical requests
- Intermediate result inspection for complex chains/agents

### Why super-openai?

When building LLM-powered chains, conversational bots, RAG pipelines or agents, `super-openai` helps in both development and production:

**In development**

- Never wait minutes when you re-run your code because all previous responses are cached in-memory.
- In complex chains or agents, quickly inspect the sequence of requests and responses to understand which response was incorrect and isolate the source of error
- When using third-party libs like `guardrails`, `langchain` or `instructor` they often modify your prompts or make additional requests under the hood, eating up tokens and adding latency. `super-openai` helps make these explicit and give you full visibility.

**In production**

- Speed up responses by serving LLM calls from cache. Currently `super-openai` only supports in-memory caching, but we'll support other cache destinations in the near future.
- Easily capture logs of openai requests and responses, including inputs/outputs, token usage, and latency. Log them to your preferred observability tool. Currently we only log openai requests, but we'll add more detailed tracing in the future.
- 100% private and secure: `super-openai` runs entirely inside your environment and never sends any data outside.

### Installation & basic usage

Run `pip install super-openai` or `poetry add super-openai`

To initialize super-openai run

```
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
Messages:
- user: What's the capital of France?
Arguments:
- model: gpt-4-1106-preview
Output:
- assistant: The capital of France is Paris.
Metadata:
- Prompt tokens: 14
- Completion tokens: 7
- Total tokens: 21
- Start time: 1708985826.1323142
- Latency: 0.8096299171447754
Cached: False
```

</details>

You can also avoid the context manager and directly manage starting and stopping loggers.

```python
logger = init_logger()
client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
  {"role": "user", "content": "What's the capital of France?"}
  ])
print(logger.logs[0])
logger.end()
```

<details>
<summary> Expand to see output</summary>

```
Messages:
- user: What's the capital of France?
Arguments:
- model: gpt-4-1106-preview
Output:
- assistant: The capital of France is Paris.
Metadata:
- Prompt tokens: 14
- Completion tokens: 7
- Total tokens: 21
- Start time: 1708985829.87182
- Latency: 8.106231689453125e-05
Cached: True
**************************************************
```

</details>

Notice the second request's latency is almost 0 and `Cached` is `True`

For advanced usage and examples see below.

## Logging

super-openai wraps the `OpenAI.chat.completions.create` and `AsyncOpenAI.chat.completions.create` functions and stores logs into a `super_openai.Logger` object. The following fields are captured and logged:

- Input prompt(s)
- Input parameters (model, temperature, etc.)
- Response(s)
- Metadata
  - Token usage (prompt and response)
  - Latency

**Getting Started**
TODO

**Statistics**
TODO

**Advanced Logging**
TODO

**Why logging?**
Logging the inputs and outputs of OpenAI calls is crucial for several reasons:

- Visibility: It provides clear insight into the actual requests and prompts being made. This transparency is key to understanding how your application interacts with OpenAI's API, ensuring that the prompts sent are as intended.

- Performance Analysis: By logging, you can monitor the cost (in terms of token usage) and the response time of your OpenAI calls. This information is vital for optimizing the efficiency of your pipelines/agents, helping you to identify bottlenecks and reduce operational costs.

- Debugging and Error Isolation: When unexpected or incorrect results occur, logs are invaluable. They allow you to trace back through the sequence of requests and responses, making it easier to pinpoint exactly where and why an error was introduced. This can significantly speed up the debugging process, saving time and effort in development and production environments.

## Caching

`super-openai` caches all requests in-memory using `cachetools` and returns the cached response next time if all request parameters are exactly the same and the same `OpenAI` client is used.

Caching is automatically enabled when you called `init_super_openai` and applies both to regular `chat.completion.create` and async `chat.completion.create` requests. It works in both streaming and regular mode.

You can disable caching or change the cache size (default 1000) when initializating super-openai:

```python
from super_openai import init_super_openai

init_super_openai(enable_caching=True, cache_size=100)
```

**Why caching?**

Caching repeated requests speeds up development. Often you end up changing one part of the pipeline and having to run the entire pipeline on the same dataset. All the requests are re-run, which wastes time and costs money. Instead, caching ensures only the parts that change actually make new requests.

Cache hits may be less frequent in production, but are still useful for improving latency and reducing cost, especially in consumer products where users can accidentally re-do the same request.

## Future work

- Log function calling and tool usage responses
- Simplifying retries
- Tracing
- Disk and remote caching
- Thread-safe caching
- Integrate with native python logging
- Integrate with 3rd party hosted logging services

## Contributing

Contributions to the Super OpenAI Python Wrapper are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

The Super OpenAI Python Wrapper is released under the MIT License. See the `LICENSE` file for more details.
