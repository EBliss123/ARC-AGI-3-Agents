class ObjectLedger:
    def __init__(self):
        self.sets = {}
        self._next_id = 1

    def add_set(self, pixels):
        """
        Registers a new Set. 
        'pixels' expects a list of individual pixel data, ensuring 
        the exact colors and coordinates of every involved pixel are stored.
        """
        set_id = f"Set_{self._next_id:03d}"
        self.sets[set_id] = pixels
        self._next_id += 1
        return set_id
        
    def get_set(self, set_id):
        """Retrieves the exact pixel data for a specific Set ID."""
        return self.sets.get(set_id)
        
    def wipe_ledger(self):
        """Instantly destroys all defined Sets to allow for theory recalculation."""
        self.sets = {}
        self._next_id = 1