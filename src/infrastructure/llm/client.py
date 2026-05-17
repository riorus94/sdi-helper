import os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package is required. Install it with: pip install openai")

from .compressor import compress_messages

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def complete(messages: list[dict]):
    final_messages = compress_messages(messages)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=final_messages,
    )

    return response.choices[0].message
