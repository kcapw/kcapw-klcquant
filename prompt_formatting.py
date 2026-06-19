from __future__ import annotations

from typing import Any


def format_prompt(tokenizer: Any, prompt: str, prompt_format: str = "raw", system_prompt: str | None = None) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format != "chat":
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_format_reference(tokenizer: Any, prompt: str, system_prompt: str | None = None) -> dict:
    raw = format_prompt(tokenizer, prompt, "raw", system_prompt)
    chat = format_prompt(tokenizer, prompt, "chat", system_prompt)
    raw_ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
    chat_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    return {
        "raw_text": raw,
        "chat_text": chat,
        "raw_token_ids": raw_ids,
        "chat_token_ids": chat_ids,
        "raw_token_count": len(raw_ids),
        "chat_token_count": len(chat_ids),
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
