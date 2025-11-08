# In agents/obml_agi_3.py

from .agent import Agent, FrameData
from .structs import GameAction, GameState

class ObmlAgi3Agent(Agent):
    """
    A bare-minimum agent.
    """
    def __init__(self, params: dict = None, **kwargs):
        super().__init__(**kwargs)
        # ... any other init you need ...

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Picks the first available action.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        
        # Just pick the first action
        if latest_frame.available_actions:
            return latest_frame.available_actions[0]
        
        # Fallback
        return GameAction.RESET

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False