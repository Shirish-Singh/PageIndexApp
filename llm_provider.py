"""
Generic LLM Provider Interface

This module provides a unified interface for multiple LLM providers.
To switch providers, simply change the LLM_PROVIDER environment variable
or call set_provider() with the desired provider name.

Supported providers:
- groq: Groq API (free tier available)
- openai: OpenAI API
- ollama: Local Ollama models

Usage:
    from llm_provider import get_llm_client, chat_completion

    # Get configured client
    client = get_llm_client()

    # Simple completion
    response = chat_completion("What is 2+2?")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator, Union
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """
    Get a secret from environment variables or Streamlit secrets.
    Supports both local development (.env) and Streamlit Cloud deployment.
    """
    # First try environment variable
    value = os.getenv(key, "")
    if value:
        return value

    # Then try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 8192


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Send chat messages and get response."""
        pass

    @abstractmethod
    def get_openai_compatible_client(self):
        """Return an OpenAI-compatible client for PageIndex integration."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider using OpenAI-compatible API."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or get_secret("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        # Use OpenAI client with Groq's endpoint
        import openai
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        model = kwargs.get("model", self.config.model or self.DEFAULT_MODEL)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return generate()

        return response.choices[0].message.content

    def get_openai_compatible_client(self):
        return self.client, self.async_client


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    DEFAULT_MODEL = "gpt-4o-2024-11-20"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or get_secret("OPENAI_API_KEY") or get_secret("CHATGPT_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or CHATGPT_API_KEY environment variable is required")

        import openai
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        model = kwargs.get("model", self.config.model or self.DEFAULT_MODEL)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return generate()

        return response.choices[0].message.content

    def get_openai_compatible_client(self):
        return self.client, self.async_client


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)

        import openai
        self.client = openai.OpenAI(
            api_key="ollama",  # Ollama doesn't require real API key
            base_url=self.base_url
        )
        self.async_client = openai.AsyncOpenAI(
            api_key="ollama",
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        model = kwargs.get("model", self.config.model or self.DEFAULT_MODEL)
        temperature = kwargs.get("temperature", self.config.temperature)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream
        )

        if stream:
            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return generate()

        return response.choices[0].message.content

    def get_openai_compatible_client(self):
        return self.client, self.async_client


# Provider registry
PROVIDERS = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}

# Global provider instance
_current_provider: Optional[BaseLLMProvider] = None
_current_config: Optional[LLMConfig] = None


def get_default_config() -> LLMConfig:
    """Get default LLM configuration from environment variables or Streamlit secrets."""
    provider = get_secret("LLM_PROVIDER", "groq").lower()

    # Provider-specific defaults
    model_defaults = {
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-2024-11-20",
        "ollama": "llama3.2",
    }

    return LLMConfig(
        provider=provider,
        model=get_secret("LLM_MODEL", model_defaults.get(provider, "")),
        api_key=get_secret(f"{provider.upper()}_API_KEY"),
        temperature=float(get_secret("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(get_secret("LLM_MAX_TOKENS", "8192")),
    )


def set_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """
    Set the active LLM provider.

    Args:
        provider_name: One of 'groq', 'openai', 'ollama'
        **kwargs: Additional config options (model, api_key, temperature, etc.)

    Returns:
        The configured provider instance

    Example:
        # Switch to Groq with custom model
        set_provider("groq", model="llama-3.1-8b-instant")

        # Switch to OpenAI
        set_provider("openai", model="gpt-4o")
    """
    global _current_provider, _current_config

    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: {list(PROVIDERS.keys())}")

    # Get model from kwargs, or from environment, or use provider default
    model = kwargs.get("model") or get_secret("LLM_MODEL", "")

    config = LLMConfig(
        provider=provider_name,
        model=model,
        api_key=kwargs.get("api_key"),
        base_url=kwargs.get("base_url"),
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 8192),
    )

    _current_config = config
    _current_provider = PROVIDERS[provider_name](config)
    return _current_provider


def get_provider() -> BaseLLMProvider:
    """Get the current LLM provider, initializing with defaults if needed."""
    global _current_provider, _current_config

    if _current_provider is None:
        config = get_default_config()
        _current_config = config
        _current_provider = PROVIDERS[config.provider](config)

    return _current_provider


def get_llm_client():
    """Get the OpenAI-compatible client for the current provider."""
    provider = get_provider()
    return provider.get_openai_compatible_client()


def chat_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[str, Generator[str, None, None]]:
    """
    Simple interface for chat completion.

    Args:
        prompt: User message
        system_prompt: Optional system message
        stream: Whether to stream the response
        **kwargs: Additional parameters (model, temperature, max_tokens)

    Returns:
        Response string or generator for streaming
    """
    provider = get_provider()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return provider.chat(messages, stream=stream, **kwargs)


def patch_pageindex_for_provider():
    """
    Monkey-patch PageIndex's utils module to use our LLM provider.
    Call this before importing PageIndex.
    """
    import sys

    # Get the OpenAI-compatible clients
    sync_client, async_client = get_llm_client()
    provider = get_provider()

    # We need to patch the openai module that PageIndex imports
    import openai

    # Create patched OpenAI class
    original_openai_class = openai.OpenAI
    original_async_openai_class = openai.AsyncOpenAI

    class PatchedOpenAI(original_openai_class):
        def __init__(self, *args, **kwargs):
            # Override with our provider's settings
            if isinstance(provider, GroqProvider):
                kwargs['api_key'] = provider.api_key
                kwargs['base_url'] = GroqProvider.BASE_URL
            elif isinstance(provider, OllamaProvider):
                kwargs['api_key'] = "ollama"
                kwargs['base_url'] = provider.base_url
            # For OpenAI provider, use original settings
            super().__init__(*args, **kwargs)

    class PatchedAsyncOpenAI(original_async_openai_class):
        def __init__(self, *args, **kwargs):
            if isinstance(provider, GroqProvider):
                kwargs['api_key'] = provider.api_key
                kwargs['base_url'] = GroqProvider.BASE_URL
            elif isinstance(provider, OllamaProvider):
                kwargs['api_key'] = "ollama"
                kwargs['base_url'] = provider.base_url
            super().__init__(*args, **kwargs)

    # Apply patches
    openai.OpenAI = PatchedOpenAI
    openai.AsyncOpenAI = PatchedAsyncOpenAI


def get_pageindex_model() -> str:
    """Get the appropriate model name for PageIndex based on current provider."""
    provider = get_provider()

    if isinstance(provider, GroqProvider):
        return provider.config.model or "llama-3.3-70b-versatile"
    elif isinstance(provider, OllamaProvider):
        return provider.config.model or "llama3.2"
    else:
        return provider.config.model or "gpt-4o-2024-11-20"


def patch_tiktoken():
    """
    Patch tiktoken to handle non-OpenAI model names.
    This is needed because PageIndex uses tiktoken.encoding_for_model()
    which doesn't recognize Groq/Ollama model names.
    """
    import tiktoken

    original_encoding_for_model = tiktoken.encoding_for_model

    def patched_encoding_for_model(model_name: str):
        try:
            return original_encoding_for_model(model_name)
        except KeyError:
            # For non-OpenAI models, use cl100k_base (GPT-4 encoding)
            # This is a reasonable approximation for most LLMs
            return tiktoken.get_encoding("cl100k_base")

    tiktoken.encoding_for_model = patched_encoding_for_model


def patch_pageindex_json_extraction():
    """
    Patch PageIndex's extract_json to be more robust with different LLM outputs.
    """
    import sys
    import json
    import re

    # We need to patch after PageIndex is imported
    def robust_extract_json(content: str) -> dict:
        """More robust JSON extraction that handles various LLM output formats."""
        if not content:
            return {}

        try:
            # First, try to extract JSON enclosed within ```json and ```
            start_idx = content.find("```json")
            if start_idx != -1:
                start_idx += 7
                end_idx = content.rfind("```")
                json_content = content[start_idx:end_idx].strip()
            else:
                # Try to find JSON object or array
                json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_content = content.strip()

            # Clean up common issues
            json_content = json_content.replace('None', 'null')
            json_content = json_content.replace("'", '"')  # Replace single quotes

            # Try to parse
            result = json.loads(json_content)

            # Ensure we have expected keys with defaults
            if isinstance(result, dict):
                if 'toc_detected' not in result and 'answer' not in result:
                    # Try to infer from content
                    content_lower = content.lower()
                    if 'yes' in content_lower:
                        result['toc_detected'] = 'yes'
                        result['answer'] = 'yes'
                    else:
                        result['toc_detected'] = 'no'
                        result['answer'] = 'no'

            return result

        except (json.JSONDecodeError, Exception) as e:
            # Last resort: try to extract yes/no from content
            content_lower = content.lower()
            if 'yes' in content_lower:
                return {'toc_detected': 'yes', 'answer': 'yes', 'thinking': content}
            else:
                return {'toc_detected': 'no', 'answer': 'no', 'thinking': content}

    return robust_extract_json


def get_available_models(provider_name: Optional[str] = None) -> List[str]:
    """Get list of available models for a provider."""
    models = {
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "openai": [
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "ollama": [
            "llama3.2",
            "llama3.1",
            "mistral",
            "mixtral",
            "phi3",
            "gemma2",
        ],
    }

    if provider_name:
        return models.get(provider_name, [])
    return models
