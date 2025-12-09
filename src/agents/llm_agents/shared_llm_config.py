"""
Shared LLM configuration/utility helpers used by GPT and Claude agents.

This module keeps all provider specific wiring (API keys, client initialization,
request helpers) in one place so that the individual agents can stay focused on
their trading logic.  The implementation is intentionally defensive: it degrades
gracefully when SDKs or API keys are missing and simply reports a structured
error back to the caller instead of crashing the whole dashboard.
"""

from __future__ import annotations

import os


import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from src.config.config import Config
from src.util.logging import get_logger


class SharedLLMConfig:
    """Utility wrapper that manages LLM providers and request execution."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.log = get_logger(name="SharedLLM", enable_console=False)
        self.providers: Dict[str, Dict[str, Any]] = {}

        # Lazily configure supported providers.
        self._init_provider(
            name="openai",
            env_var="OPENAI_API_KEY",
            factory=self._create_openai_client,
        )
        self._init_provider(
            name="anthropic",
            env_var="ANTHROPIC_API_KEY",
            factory=self._create_anthropic_client,
        )
        self._init_provider(
            name="deepseek",
            env_var="DEEPSEEK_API_KEY",
            factory=self._create_deepseek_client,
        )

    
    
    def _init_provider(self, name: str, env_var: str, factory):
        load_dotenv()
        time.sleep(0.5)
        
        cfg = self.config.get(f"llm.{name}", {}) or {}
        
        api_key = os.getenv(env_var)
        if not env_var:
            self.log.error("ENV FILE NOT FOUND PLEASE SEE")
        enabled = bool(cfg.get("enabled")) and bool(api_key)

        client = None
        if enabled:
            client = factory(api_key)
            if client is None:
                enabled = False

        self.providers[name] = {
            "config": cfg,
            "api_key": api_key,
            "client": client,
            "available": enabled,
        }

        if not enabled:
            self.log.info(
                f"{name.title()} provider disabled or not fully configured; "
                "requests will return a friendly error."
            )

    def is_available(self, provider: str) -> bool:
        """Return True when the provider has a client and API key configured."""
        return bool(self.providers.get(provider, {}).get("available"))

    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Expose the raw provider configuration for agents."""
        return self.providers.get(provider, {}).get("config", {}) or {}
    
    def generate(
        self,
        prompt: str,
        provider: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a chat completion style request against the requested provider.

        Returns a dict with at least `response` on success or `error` on failure.
        """
        if not self.is_available(provider):
            return {"error": f"{provider} provider not available. Check API keys and configuration."}

        cfg = self.get_model_config(provider)
        final_temperature = temperature if temperature is not None else cfg.get("temperature", 0.7)
        final_max_tokens = max_tokens or cfg.get("max_tokens", 1000)
        final_system_prompt = system_prompt or cfg.get("system_prompt", "You are a helpful trading assistant.")

        try:
            if provider == "openai":
                return self._generate_openai(prompt, final_system_prompt, final_temperature, final_max_tokens, cfg)
            if provider == "anthropic":
                return self._generate_anthropic(prompt, final_system_prompt, final_temperature, final_max_tokens, cfg)
            if provider == "deepseek":
                return self._generate_deepseek(prompt, final_system_prompt, final_temperature, final_max_tokens, cfg)
        except Exception as exc:
            self.log.error(f"{provider.title()} request failed: {exc}", exc_info=True)
            return {"error": str(exc)}

        return {"error": f"Unsupported provider: {provider}"}

    # ------------------------------------------------------------------ #
    # Provider specific implementations
    # ------------------------------------------------------------------ #
    def _create_openai_client(self, api_key: str):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.log.warning("openai package not installed; GPT agent disabled.")
            return None

        try:
            return OpenAI(api_key=api_key)
        except Exception as exc:
            self.log.error(f"Failed to initialize OpenAI client: {exc}", exc_info=True)
            return None

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self.providers["openai"]["client"]
        if client is None:
            return {"error": "OpenAI client not initialized"}

        model_name = cfg.get("model", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message.get("content", "")
        usage = getattr(response, "usage", None) or {}
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)),
            "completion_tokens": getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)),
            "total_tokens": getattr(usage, "total_tokens", usage.get("total_tokens", 0)),
        }

        return {
            "response": message,
            "usage": usage_dict,
            "cost": self._estimate_cost("openai", usage_dict),
        }

    def _create_anthropic_client(self, api_key: str):
        try:
            from anthropic import Anthropic
        except ImportError:
            self.log.warning("anthropic package not installed; Claude agent disabled.")
            return None

        try:
            return Anthropic(api_key=api_key)
        except Exception as exc:
            self.log.error(f"Failed to initialize Anthropic client: {exc}", exc_info=True)
            return None

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self.providers["anthropic"]["client"]
        if client is None:
            return {"error": "Anthropic client not initialized"}

        model_name = cfg.get("model", "claude-3-5-sonnet-20241022")
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        # Claude responses return a sequence of content blocks.
        content_text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
        usage = getattr(response, "usage", None)

        # Handle both object and dict formats for usage
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) if hasattr(usage, "input_tokens") else usage.get("input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0) if hasattr(usage, "output_tokens") else usage.get("output_tokens", 0)
        else:
            input_tokens = 0
            output_tokens = 0

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        return {
            "response": content_text,
            "usage": usage_dict,
            "cost": self._estimate_cost("anthropic", usage_dict),
        }

    def _create_deepseek_client(self, api_key: str):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.log.warning("openai package not installed; DeepSeek agent disabled.")
            return None

        try:
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        except Exception as exc:
            self.log.error(f"Failed to initialize DeepSeek client: {exc}", exc_info=True)
            return None

    def _generate_deepseek(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self.providers["deepseek"]["client"]
        if client is None:
            return {"error": "DeepSeek client not initialized"}

        model_name = cfg.get("model", "deepseek-chat")

        # DeepSeek Reasoner doesn't support system prompts in the traditional way
        # We'll prepend the system prompt to the user message
        if "reasoner" in model_name.lower():
            full_prompt = f"{system_prompt}\n\n{prompt}"
            messages = [{"role": "user", "content": full_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message.get("content", "")
        usage = getattr(response, "usage", None) or {}

        # DeepSeek Reasoner includes reasoning tokens
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)),
            "completion_tokens": getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)),
            "reasoning_tokens": getattr(usage, "prompt_cache_hit_tokens", usage.get("prompt_cache_hit_tokens", 0)),
            "total_tokens": getattr(usage, "total_tokens", usage.get("total_tokens", 0)),
        }

        # Extract reasoning content if available (for reasoner model)
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)

        result = {
            "response": message,
            "usage": usage_dict,
            "cost": self._estimate_cost("deepseek", usage_dict),
        }

        if reasoning_content:
            result["reasoning_process"] = reasoning_content

        return result

    # ------------------------------------------------------------------ #
    def _estimate_cost(self, provider: str, usage: Dict[str, int]) -> float:
        """
        Placeholder for cost estimation.  We keep the hook so the dashboard can
        display something sensible even when cost tracking is not configured.
        """
        if not usage:
            return 0.0

        # DeepSeek Reasoner has different pricing for reasoning vs completion tokens
        if provider == "deepseek" and usage.get("reasoning_tokens"):
            # DeepSeek Reasoner pricing (as of Jan 2025):
            # Input: $0.55 per 1M tokens
            # Reasoning: $0.55 per 1M tokens (cached) / $2.19 per 1M tokens (uncached)
            # Output: $2.19 per 1M tokens
            prompt_tokens = usage.get("prompt_tokens", 0)
            reasoning_tokens = usage.get("reasoning_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            prompt_cost = (prompt_tokens / 1_000_000) * 0.55
            reasoning_cost = (reasoning_tokens / 1_000_000) * 0.55  # Assuming cached
            completion_cost = (completion_tokens / 1_000_000) * 2.19

            return round(prompt_cost + reasoning_cost + completion_cost, 6)

        # Basic token based approximation for other providers
        token_count = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        if token_count is None:
            return 0.0

        # Default rough pricing in USD per 1K tokens (updated when real prices available).
        pricing = {
            "openai": 0.005,
            "anthropic": 0.008,
            "deepseek": 0.0014,  # DeepSeek Chat is typically cheaper
        }
        rate = pricing.get(provider, 0.0)
        return round((token_count / 1000.0) * rate, 6)
