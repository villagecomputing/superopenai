import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

# Cost per 1M tokens
_cost_dict = {
    "gpt-4-0125-preview": [10, 30],
    "gpt-4-turbo-preview": [10, 30],
    "gpt-4-1106-preview": [10, 30],
    "gpt-4-1106-vision-preview": [10, 30],
    "gpt-4-vision-preview": [10, 30],
    "gpt-4": [30, 60],
    "gpt-4-0613": [30, 60],
    "gpt-4-32k": [60, 120],
    "gpt-4-32k-0613": [60, 120],
    "gpt-3.5-turbo-0125": [0.5, 1.5],
    "gpt-3.5-turbo": [0.5, 1.5],
    "gpt-3.5-turbo-1106": [0.5, 1.5],
    "gpt-3.5-turbo-instruct": [1.5, 2],
}


def update_cost_dict(cost_dict):
    global _cost_dict
    _cost_dict.update(cost_dict)


def get_cost(prompt_tokens, completion_tokens, model):
    """
    Return the cost of a completion based on the number of tokens in the prompt and completion.
    """
    cost_dict = _cost_dict.get(model)
    if not cost_dict or not prompt_tokens or not completion_tokens:
        return None
    return (cost_dict[0]*prompt_tokens + cost_dict[1]*completion_tokens)/1e6


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Return the number of tokens used by a list of messages.
    From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print(
        # "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # NOTE: added str() - hack for handling tool calls, otherwise it errors
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
