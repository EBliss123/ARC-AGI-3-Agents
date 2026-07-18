class PhysicsRulebook:
    def __init__(self):
        self.baseline_rules = []
        self.compressed_rules = []

    def set_baseline(self, rules):
        """Sets the raw, bloated pixel-level truth."""
        self.baseline_rules = rules

    def set_compressed(self, rules):
        """Sets the smaller, abstracted rules chosen by the network."""
        self.compressed_rules = rules

    def get_active_rules(self):
        """Returns compressed rules if they exist, otherwise falls back to baseline."""
        return self.compressed_rules if self.compressed_rules else self.baseline_rules