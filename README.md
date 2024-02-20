`super-openai` gives your openai client superpowers with improved logging and request/response caching.

It aims to solve the following problems:

- Prompt and request visibility
- Token usage and latency counting across multiple requests
- Caching repeated identical requests
- Intermediate result inspection for complex chains/agents

### Installation & Usage

Installation is as simple as `pip install super-openai` with `pip` or `poetry add super-openai` with `poetry`.

To start using `super-openai` call `init_super_openai` - this will monkey-patch the relevant functions on the `OpenAI` object. Then you can use the `openai` library as usual with all the superpowers of `super-openai`

```
from super_openai import init_super_openai

init_super_openai()
```

**Basic logging example**

```
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

This would print

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

You can also avoid the context manager and directly manage starting and stopping loggers.

```
logger = init_logger()
client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
  {"role": "user", "content": "What's the capital of France?"}
  ])
print(logger.logs[0])
logger.end()
```

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

Notice the second request's latency is almost 0 and `Cached` is `True`

### Why super-openai?

When building LLM-powered chains, conversational bots, RAG pipelines or agents, `super-openai` helps in both development and production:

In development:

- Never wait minutes when you re-run your code because all previous responses are cached in-memory.
- In complex chains or agents, quickly inspect the sequence of requests and responses to understand which response was incorrect and isolate the source of error
- When using third-party libs like `guardrails`, `langchain` or `instructor` they often modify your prompts or make additional requests under the hood, eating up tokens and adding latency. `super-openai` helps make these explicit and give you full visibility.

Production use cases:

- Speed up responses by serving LLM calls from cache. Currently `super-openai` only supports in-memory caching, but we'll support other cache destinations in the near future.
- Easily capture logs of openai requests and responses, including inputs/outputs, token usage, and latency. Log them to your preferred observability tool. Currently we only log openai requests, but we'll add more detailed tracing in the future.
- 100% private and secure: `super-openai` runs entirely inside your environment and never sends any data outside.

### Logging

### Caching

### Future work

- Simplifying retries
- Tracing
- Disk and remote caching
-

## Contributing

Contributions to the Super OpenAI Python Wrapper are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

The Super OpenAI Python Wrapper is released under the MIT License. See the `LICENSE` file for more details.
