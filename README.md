_Logging and caching superpowers for your openai client_

## Introduction

super-openai was built to solve the following problems:

**Prompt and request visibility**. Many popular libraries like `langchain`, `guardrails`, `instructor` modify your prompts or even make additional requests under the hood. Sometimes this is useful, sometimes it's counter-productive. We believe it's good to adopt a "[show me the prompt](https://hamel.dev/blog/posts/prompt/)" attitude.

**Cost and latency tracking**. How much did my requests cost? How long did they take? These are important factors that affect the quality and feasibility of your software. Especially when you're chaining multiple LLM calls or building complex agent flows, it's important keep track of performance and understand which step is the bottleneck. This is useful both in development and production.

**Repeated identical requests**. We often find ourselves re-running the same pipeline with a minor edit and having to re-execute every LLM call to see the new output. This unnecessarily slows down the development and iteration cycle. Also, in production this is a missed opportuntity to provide faster responses and reduce cost.

**Debugging complex chains/agents**. Complex chains or agents go wrong because of cascading failure. To debug the failure we need to inspect intermediate results and identify the source of error, then improve the prompt, try a different model, etc. It starts with logging and eyeballing the sequence of requests.

**Privacy, security and speed**. Many great tools exist to help you solve the above by sending your data to a remote server. But sometimes you need the data to stay local. Other times, you want to do some quick and dirty development without having to set up an api key, sign up for a service, understand their library and UI.

### Installation & basic usage

Run `pip install super-openai` or `poetry add super-openai`

To initialize super-openai run

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
for log in logger.logs:
  print(log)
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
