"""
LLM trading agent package.

Expose the concrete GPT, Claude, and DeepSeek agents while keeping imports lightweight.
"""

from .gpt_agent import GPTAgent
from .claude_agent import ClaudeAgent
from .deepseek_agent import DeepSeekAgent

__all__ = [
    "GPTAgent",
    "ClaudeAgent",
    "DeepSeekAgent",
]
