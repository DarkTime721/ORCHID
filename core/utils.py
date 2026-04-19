import json
import tiktoken

encoder = tiktoken.get_encoding('cl100k_base')  # Standard encoding used as Ollama does not reliably return token count/tokens used


def count_message_tokens(messages: list) -> int:
    total_toks = 0
    for m in messages:
        if hasattr(m, 'content') and isinstance(m.content, str):
            total_toks += len(encoder.encode(m.content))
    return total_toks


def append_to_json(filepath: str, new_data: dict):
    data = {k: v for k, v in new_data.items() if k != 'messages'}
    with open(filepath, 'a') as f:
        f.write(json.dumps(data, default=str) + '\n')
