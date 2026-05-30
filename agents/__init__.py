<<<<<<< HEAD
from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.multimodal import MultiModalLLM
from .templates.openclaw_agent import OpenClaw
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent

load_dotenv()
=======
from typing import Type
from .agent import Agent
from .swarm import Swarm # <--- ADD THIS LINE
from .obml_agi_3 import ObmlAgi3Agent
>>>>>>> bb87cc866110c1b1c067f72864d95ec77baba68f

# Registry for the fast_runner.py to find your agent
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "obmlagi3agent": ObmlAgi3Agent
}

__all__ = [
    "Agent",
    "Swarm", # <--- ADD THIS LINE
    "AVAILABLE_AGENTS",
<<<<<<< HEAD
    "MultiModalLLM",
    "OpenClaw",
]
=======
]
>>>>>>> bb87cc866110c1b1c067f72864d95ec77baba68f
