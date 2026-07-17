"""
Purpose: The Immutable Log.
Responsibilities: Stores every raw tensor transition. Never compressed or deleted.
"""
class ExperienceBuffer:
    def __init__(self):
        self.history = []

    def add_transition(self, transition_tensor):
        """Saves the raw mathematical reality of a frame transition."""
        self.history.append(transition_tensor)

    def get_full_history(self):
        """Returns all raw transitions so the agent can recalculate theories."""
        return self.history