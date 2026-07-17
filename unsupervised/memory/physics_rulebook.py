class PhysicsRulebook:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule_json):
        """Saves a mathematical equation or JSON rule to the working theory."""
        self.rules.append(rule_json)

    def get_rules(self):
        """Retrieves all rules in the current working theory."""
        return self.rules

    def wipe_rules(self):
        """Instantly deletes all rules so the agent is forced to look back at the raw Immutable Log."""
        self.rules = []