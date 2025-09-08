import random
from .agent import Agent, FrameData
from .structs import GameAction, GameState

class ObrlAgi3Agent(Agent):
    """
    An agent for the ARC-AGI-3 challenge using object-based reinforcement learning.
    """

    def __init__(self, **kwargs):
        """
        The constructor for the agent.
        """
        super().__init__(**kwargs)
        # A flag to make sure we only print the actions once.
        self.actions_printed = False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        This method is called by the game to get the next action.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.actions_printed = False  # Reset the print flag for the new game.
            return GameAction.RESET

        # This is the REAL list of actions for this specific game on this turn.
        game_specific_actions = latest_frame.available_actions

        # If we just discovered the game-specific actions, print them once.
        if game_specific_actions and not self.actions_printed:
            print(f"Discovered game-specific actions: {game_specific_actions}")
            self.actions_printed = True

        # If the game-specific list is available, choose from it.
        if game_specific_actions:
            return random.choice(game_specific_actions)

        # FALLBACK: If the specific list is empty (e.g., first turn),
        # choose from the master list of all actions to avoid a crash.
        fallback_options = [a for a in GameAction if a is not GameAction.RESET]
        return random.choice(fallback_options)


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """
        This method is called by the game to see if the agent thinks it is done.
        """
        return False