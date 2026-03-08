"""Shared LLM service for OpenRouter/OpenAI compatible calls."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)?$")


class LLMService:
    """Lightweight wrapper for OpenRouter chat completions."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    @staticmethod
    def is_model_name_valid(model_name: str) -> bool:
        return bool(MODEL_NAME_PATTERN.match(model_name))

    @staticmethod
    def log_backend(
        message: str,
        *args: object,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        prefix = f"[session={session_id or 'n/a'} run={run_id or 'n/a'}] "
        logger.warning(prefix + message, *args)

    def complete(
        self,
        *,
        request_name: str,
        model: str,
        max_tokens: int,
        temperature: float,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[Any, Optional[int]]:
        model_valid = self.is_model_name_valid(model)
        self.log_backend(
            "[%s] OpenRouter request start model=%s valid_model_format=%s max_tokens=%s temperature=%s messages=%s",
            request_name,
            model,
            model_valid,
            max_tokens,
            temperature,
            len(messages),
            session_id=session_id,
            run_id=run_id,
        )
        if not model_valid:
            self.log_backend(
                "[%s] Model name may be malformed: %s",
                request_name,
                model,
                session_id=session_id,
                run_id=run_id,
            )

        try:
            raw_client = self.client.chat.completions.with_raw_response
            raw_response = raw_client.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            status_code = getattr(raw_response, "status_code", None)
            self.log_backend(
                "[%s] OpenRouter response received status_code=%s",
                request_name,
                status_code,
                session_id=session_id,
                run_id=run_id,
            )
            return raw_response.parse(), status_code
        except AttributeError:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            self.log_backend(
                "[%s] OpenRouter response received status_code=unavailable",
                request_name,
                session_id=session_id,
                run_id=run_id,
            )
            return response, None
        except Exception:
            logger.exception(
                "[session=%s run=%s] [%s] OpenRouter request failed",
                session_id or "n/a",
                run_id or "n/a",
                request_name,
            )
            raise

    @staticmethod
    def extract_message_text(response: Any) -> str:
        return extract_message_text(response)


def create_openrouter_client(
    api_key: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """Create an OpenRouter client using the OpenAI-compatible API."""
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or {})


def build_prompt(template: str, user_input: str, **kwargs: object) -> str:
    """Build the LLM prompt from a template."""
    return template.format(user_input=user_input, **kwargs)


def extract_message_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible response."""
    try:
        return response.choices[0].message.content
    except AttributeError:
        return response["choices"][0]["message"]["content"]
