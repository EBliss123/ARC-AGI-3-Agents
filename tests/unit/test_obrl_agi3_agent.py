import pytest

from agents.obrl_agi3 import ObrlAgi3Agent
from agents.structs import FrameData, GameAction, GameState


def make_agent():
    agent = ObrlAgi3Agent(card_id="test-card", game_id="test-game", agent_name="obrl-agi3", ROOT_URL="https://example.com", record=False)
    # Silence debug
    for k in agent.debug_channels.keys():
        agent.debug_channels[k] = False
    return agent


def make_frame(state=GameState.NOT_FINISHED, score=0, actions=None):
    fd = FrameData(game_id="test-game", frame=[], state=state, score=score)
    if actions is None:
        actions = [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION6]
    # Provide available actions as real GameAction enum members
    fd.available_actions = actions
    return fd


@pytest.mark.unit
class TestObrlAgi3AgentBasics:
    def test_reset_on_not_played(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_PLAYED)

        action = agent.choose_action([frame], frame)
        assert action == GameAction.RESET

    def test_reset_on_game_over(self):
        agent = make_agent()
        frame = make_frame(state=GameState.GAME_OVER)

        action = agent.choose_action([frame], frame)
        assert action == GameAction.RESET

    def test_is_done_always_false(self):
        agent = make_agent()
        frame = make_frame(state=GameState.WIN)
        assert agent.is_done([frame], frame) is False


@pytest.mark.unit
class TestObrlAgi3AgentActionSelection:
    def _stub_perception_minimal(self, agent, objects):
        # Stub internal perception/analysis to deterministic outputs
        agent._perceive_objects = lambda latest_frame: objects
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {})
        agent._analyze_alignments = lambda summary: {}
        agent._analyze_diagonal_alignments = lambda summary: {}
        agent._analyze_conjunctions = lambda rels, aligns: {}
        agent._log_changes = lambda prev_summary, curr_summary: ([], curr_summary)
        agent._extract_features = lambda summary, move: {}
        agent._calculate_goal_bonus = lambda move, summary, unmet, ctx: 0.0
        agent._get_state_key = lambda summary: (len(summary),)

    def test_non_reset_on_active_state(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED)

        # Provide minimal objects so the agent has something to click or act upon
        objects = [
            {"position": (1, 2), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 11, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        action = agent.choose_action([frame], frame)
        assert action != GameAction.RESET
        assert isinstance(action, GameAction)

    def test_click_sets_coordinates_for_action6(self):
        agent = make_agent()
        # Ensure ACTION6 present and a non-click for alternative
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.RESET, GameAction.ACTION1, GameAction.ACTION6])

        objects = [
            {"position": (3, 5), "id": "obj_1", "color": 2, "size": (1, 1), "fingerprint": 7, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        action = agent.choose_action([frame], frame)

        # It may choose ACTION1 or click depending on scoring. If it chose click, coordinates must be set
        if action == GameAction.ACTION6:
            assert hasattr(action, "action_data")
            assert action.action_data.x == 5
            assert action.action_data.y == 3
        else:
            # Still a valid selection path
            assert action == GameAction.ACTION1

    def test_skip_blacklisted_action_key(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (0, 0), "id": "obj_42", "color": 3, "size": (1, 1), "fingerprint": 9, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        # Blacklist the click on this object
        blacklisted_key = agent._get_learning_key("ACTION6", "obj_42")
        agent.failed_action_blacklist.add(blacklisted_key)

        action = agent.choose_action([frame], frame)
        # Should avoid the blacklisted click and choose ACTION1
        assert action == GameAction.ACTION1

    def test_apply_failure_pattern_penalty_prefers_alternative(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (1, 1), "id": "obj_1", "color": 2, "size": (1, 1), "fingerprint": 3, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        # Simulate known failure pattern for click on obj_1 with wildcard adjacency match
        action_key = agent._get_learning_key("ACTION6", "obj_1")
        agent.failure_patterns[action_key] = {
            "adj": {"obj_1": ("x", "x", "x", "x")},
        }

        action = agent.choose_action([frame], frame)
        # With heavy penalty, it should choose ACTION1 instead of click
        assert action == GameAction.ACTION1

    def test_actions_printed_flag_only_once(self):
        agent = make_agent()
        # Keep ACTION6 available to generate possible moves
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (2, 2), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 1, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        assert agent.actions_printed is False
        _ = agent.choose_action([frame], frame)
        assert agent.actions_printed is True
        # Second call should keep it True (printed only once)
        _ = agent.choose_action([frame], frame)
        assert agent.actions_printed is True

    def test_clear_failure_blacklist_on_success_changes(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        # Prepare object and stub with a change on subsequent call
        objects_initial = [
            {"position": (0, 1), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 10, "pixels": 1},
        ]
        objects_after = [
            {"position": (0, 2), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 10, "pixels": 1},
        ]

        # First turn: no changes, establish state
        self._stub_perception_minimal(agent, objects_initial)
        _ = agent.choose_action([frame], frame)

        # Blacklist an action key
        action_key = agent._get_learning_key("ACTION6", "obj_1")
        agent.failed_action_blacklist.add(action_key)
        assert action_key in agent.failed_action_blacklist

        # Now simulate a turn with changes: stub _log_changes to report a change and updated summary
        def log_changes(prev_summary, curr_summary):
            return (["- MOVED: Object id_1 moved from (0, 1) to (0, 2)."], objects_after)

        agent._perceive_objects = lambda latest_frame: objects_after
        agent._log_changes = log_changes
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {})
        agent._analyze_alignments = lambda summary: {}
        agent._analyze_diagonal_alignments = lambda summary: {}
        agent._analyze_conjunctions = lambda rels, aligns: {}
        agent._extract_features = lambda summary, move: {}
        agent._calculate_goal_bonus = lambda move, summary, unmet, ctx: 0.0
        agent._get_state_key = lambda summary: (len(summary),)

        # Perform next action; changes should clear blacklist internally
        _ = agent.choose_action([frame], frame)
        assert len(agent.failed_action_blacklist) == 0

    def test_last_action_context_is_stored(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (4, 4), "id": "obj_1", "color": 2, "size": (1, 1), "fingerprint": 5, "pixels": 1},
        ]
        self._stub_perception_minimal(agent, objects)

        action = agent.choose_action([frame], frame)
        assert agent.last_action_context is not None
        action_name, target_id = agent.last_action_context
        assert action_name in (GameAction.ACTION6.name, GameAction.ACTION1.name)
        if action == GameAction.ACTION6:
            assert target_id == "obj_1"
        else:
            assert target_id is None
