from typing import Type
from .agent import Agent
from .obml_agi_3 import ObmlAgi3Agent

# Registry for the fast_runner.py to find your agent
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "obmlagi3agent": ObmlAgi3Agent
}

__all__ = [
    "Agent",
    "AVAILABLE_AGENTS",
]