import time
import os
import json
import random
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

# -------------------------
# Provider interfaces
# -------------------------

class ModelProvider(ABC):
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Return the model's text response for the given prompt."""
        pass

    # Optional: give a friendly name if the provider implements it
    @property
    def name(self) -> str:
        return self.__class__.__name__


class GPTProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.2,
                 max_tokens: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._name = "gpt"

        # If you want to override via env var:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return self._name

    def call(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("OpenAI package is required for GPTProvider") from e

        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set API key or OPENAI_API_KEY env var.")

        openai.api_key = self.api_key
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content.strip()


class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-2.0", temperature: float = 0.2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._name = "anthropic"
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def name(self) -> str:
        return self._name

    def call(self, prompt: str) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("anthropic package is required for AnthropicProvider") from e

        if not self.api_key:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY env var.")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Note: Anthropic API expects a specific prompt format. This is a simple wrapper.
        # You may need to adapt to the exact API usage in your environment.
        # Example payload shape (may vary by SDK version):
        resp = client.completions(
            prompt=prompt,
            model=self.model,
            max_tokens_to_sample=1024,
            temperature=self.temperature
        )
        # The exact field name may differ by SDK version; adjust as needed.
        # Some SDKs expose: resp["completion"] or resp.completion
        result = None
        if isinstance(resp, dict) and "completion" in resp:
            result = resp["completion"]
        else:
            # Fallback for some SDKs
            result = getattr(resp, "completion", "")
        return str(result).strip()


class DeepSeekProvider(ModelProvider):
    """
    A generic REST-based LLM provider. Configure endpoint and headers; it should return
    a JSON payload with a textual 'completion' or 'response' field.
    """
    def __init__(self, endpoint: str, api_key: Optional[str] = None, model: str = "default"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self._name = "deepseek"

    @property
    def name(self) -> str:
        return self._name

    def call(self, prompt: str) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "prompt": prompt}
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Expect either "completion" or "response"
        return str(data.get("completion") or data.get("response") or "")


# -------------------------
# Simple stock data helper (synthetic)
# -------------------------

def generate_synthetic_stock_history(days: int = 120, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    history = []
    price = 100.0
    date = time.time()
    for i in range(days):
        # simple random walk
        delta = random.normalvariate(0.0, 1.0)
        price = max(1.0, price + delta)
        date_i = time.strftime("%Y-%m-%d", time.localtime(date - (days - i) * 86400))
        history.append({"date": date_i, "close": round(price, 2)})
    return history


def split_data(data: List[Dict], train_frac: float = 0.7, val_frac: float = 0.2, test_frac: float = 0.1):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    n = len(data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


def build_stock_prompt(history_slice: List[Dict], target_horizon_days: int = 1) -> str:
    """
    Builds a prompt embedding recent stock data for the model to predict the next-day close.
    history_slice: Most recent days (e.g., 60 days)
    """
    lines = [f"{d['date']}: close={d['close']}" for d in history_slice]
    data_text = "\n".join(lines)
    prompt = (
        "You are a stock market analyst. Based on the recent daily close prices below, "
        f"predict the next day's close and provide a short rationale.\n\n{data_text}\n\n"
        "Question: What is the expected next-day closing price and the direction (up/down) based on this data?"
    )
    return prompt


# -------------------------
# The general llmwrapper
# -------------------------

class llmwrapper:
    def __init__(self,
                 providers: List[ModelProvider],
                 system_prompt: str = "",
                 prompt_template: Optional[str] = None,
                 memory_size: int = 512,
                 combine: str = "concat",
                 verbose: bool = False):
        self.providers = list(providers)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template or "{system}\n{user_input}"
        self.memory: List[str] = []
        self.memory_size = memory_size
        self.combine = combine  # how to ensemble: "concat" or "mean" (for numeric results)
        self.verbose = verbose
        self.log: List[str] = []

    def add_provider(self, p: ModelProvider):
        self.providers.append(p)

    def _log(self, msg: str):
        self.log.append(msg)
        if self.verbose:
            print("[llmwrapper]", msg)

    def _build_prompt(self, user_input: str) -> str:
        mem = "\n".join(self.memory[-self.memory_size:]) if self.memory else ""
        prompt = self.prompt_template.format(system=self.system_prompt, user_input=user_input)
        if mem:
            prompt += f"\n\nMemory:\n{mem}"
        return prompt

    def run(self, user_input: str) -> Dict[str, str]:
        """
        Run the given user_input across all providers. Returns a dict: {provider_name: response}
        """
        results: Dict[str, str] = {}
        prompt = user_input  # you can also prepend system prompt if needed

        # If you want to inject the history into the prompt, uncomment:
        # prompt = self._build_prompt(user_input)

        t0 = time.time()
        for p in self.providers:
            name = getattr(p, "name", p.__class__.__name__)
            try:
                if self.verbose:
                    self._log(f"Calling provider '{name}'")
                resp = p.call(prompt)
                results[name] = resp
                self._log(f"Provider '{name}' returned {len(resp)} chars")
            except Exception as e:
                results[name] = f"Error: {e}"
                self._log(f"Provider '{name}' error: {e}")
        t1 = time.time()
        if self.verbose:
            self._log(f"Total cross-provider call time: {t1 - t0:.2f}s")
        # Update memory with user input and raw responses (optional)
        self.memory.extend([f"User: {user_input}"])
        for k, v in results.items():
            self.memory.append(f"{k}: {v[:200]}")
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
        return results

    def ensemble(self, responses: Dict[str, str]) -> str:
        """
        Simple ensemble strategy:
        - Try to parse numeric outputs and average them
        - Otherwise concatenate with provider tags
        """
        nums = []
        for name, text in responses.items():
            try:
                # Try to extract a plain number from the text
                val = float(text.strip().split()[0])
                nums.append(val)
            except Exception:
                continue

        if nums:
            mean_val = sum(nums) / len(nums)
            return f"Ensembled numeric prediction (mean): {mean_val:.4f}"
        else:
            parts = [f"[{name}] {text}" for name, text in responses.items()]
            return "Ensembled responses:\n" + "\n".join(parts)


# -------------------------
# Demo / usage (stock-prediction)
# -------------------------

if __name__ == "__main__":
    # Build providers (adjust API keys/env vars as needed)
    gpt = GPTProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.2
    )

    anthro = AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-2.0",
        temperature=0.2
    )

    # DeepSeek API endpoint (replace with real URL)
    deepseek = DeepSeekProvider(
        endpoint="https://api.deepseek.ai/v1/llm/generate",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        model="default"
    )

    wrapper = llmwrapper(
        providers=[gpt, anthro, deepseek],
        system_prompt="You are a data-driven stock market analyst.",
        prompt_template="{system}\n\n{user_input}",
        memory_size=512,
        combine="concat",
        verbose=True
    )

    # Generate synthetic stock history
    history = generate_synthetic_stock_history(days=120, seed=123)

    # 70/20/10 split (train/val/test) to illustrate data partitioning
    train, val, test = split_data(history, train_frac=0.7, val_frac=0.2, test_frac=0.1)

    # Use the most recent 60 days from train+val as context for the prompt
    ctx_days = 60
    recent_context = history[-ctx_days:] if len(history) >= ctx_days else history

    # Build a stock-prediction prompt embedding the recent data
    prompt = build_stock_prompt(recent_context, target_horizon_days=1)

    # Run across providers
    results = wrapper.run(prompt)

    # Ensemble (average if numeric else concatenate)
    final = wrapper.ensemble(results)

    print("Individual provider results:")
    for k, v in results.items():
        print(f"- {k}: {v[:300]}")

    print("\nEnsembled result:")
    print(final)