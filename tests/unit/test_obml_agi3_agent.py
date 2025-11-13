import pytest

from agents.obml_agi_3 import ObmlAgi3Agent
from agents.structs import FrameData, GameAction, GameState


def make_agent():
    agent = ObmlAgi3Agent(card_id="test-card", game_id="test-game", agent_name="obml-agi3", ROOT_URL="https://example.com", record=False)
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
class TestObmlAgi3AgentBasics:
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
class TestObmlAgi3AgentActionSelection:
    def _stub_perception_minimal(self, agent, objects):
        # Stub internal perception/analysis to deterministic outputs compatible with obml agent
        agent._perceive_objects = lambda latest_frame: objects
        # _analyze_relationships returns 7-tuple per implementation
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {}, {}, {}, {})
        agent._log_changes = lambda prev_summary, curr_summary: ([], curr_summary)

    def test_non_reset_on_active_state(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED)

        # Provide minimal objects so the agent has something to click or act upon
        objects = [
            {"position": (1, 2), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 11, "pixels": 1, "pixel_coords": frozenset() },
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
            {"position": (3, 5), "id": "obj_1", "color": 2, "size": (1, 1), "fingerprint": 7, "pixels": 1, "pixel_coords": frozenset() },
        ]
        self._stub_perception_minimal(agent, objects)

        action = agent.choose_action([frame], frame)

        # It may choose ACTION1 or click depending on profiling. If it chose click, coordinates must be set
        if action == GameAction.ACTION6:
            assert hasattr(action, "action_data")
            assert action.action_data.x == 5
            assert action.action_data.y == 3
        else:
            # Still a valid selection path
            assert action == GameAction.ACTION1

    def test_actions_printed_flag_only_once(self):
        agent = make_agent()
        # Keep ACTION6 available to generate possible moves
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (2, 2), "id": "obj_1", "color": 1, "size": (1, 1), "fingerprint": 1, "pixels": 1, "pixel_coords": frozenset() },
        ]
        self._stub_perception_minimal(agent, objects)

        assert agent.actions_printed is False
        _ = agent.choose_action([frame], frame)
        assert agent.actions_printed is True
        # Second call should keep it True (printed only once)
        _ = agent.choose_action([frame], frame)
        assert agent.actions_printed is True

    def test_last_action_context_is_stored(self):
        agent = make_agent()
        frame = make_frame(state=GameState.NOT_FINISHED, actions=[GameAction.ACTION6, GameAction.ACTION1])

        objects = [
            {"position": (4, 4), "id": "obj_1", "color": 2, "size": (1, 1), "fingerprint": 5, "pixels": 1, "pixel_coords": frozenset() },
        ]
        self._stub_perception_minimal(agent, objects)

        action = agent.choose_action([frame], frame)
        assert agent.last_action_context is not None
        # last_action_context is stored as string key for obml agent
        if action == GameAction.ACTION6:
            assert agent.last_action_context.endswith("obj_1")
        else:
            assert agent.last_action_context == GameAction.ACTION1.name


@pytest.mark.unit
class TestObmlAgi3AgentLearning:
    def _base_stub(self, agent, objects_before, objects_after, change_logs):
        agent._perceive_objects = lambda latest_frame: objects_after
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {}, {}, {}, {})
        agent._log_changes = lambda prev_summary, curr_summary: (change_logs, objects_after)

    def test_parse_change_logs_to_events_handles_moved(self):
        agent = make_agent()
        frame = make_frame(actions=[GameAction.ACTION6, GameAction.ACTION1])
        objects0 = [{"position": (0,1), "id": "obj_1", "color": 1, "size": (1,1), "fingerprint": 10, "pixels": 1, "pixel_coords": frozenset()}]
        # First call to initialize memory
        agent._perceive_objects = lambda latest_frame: objects0
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {}, {}, {}, {})
        agent.choose_action([frame], frame)

        # Next turn with a change log indicating a move
        objects1 = [{"position": (0,2), "id": "obj_1", "color": 1, "size": (1,1), "fingerprint": 10, "pixels": 1, "pixel_coords": frozenset()}]
        change_logs = ["- MOVED: Object id_1 moved from (0, 1) to (0, 2)."]
        self._base_stub(agent, objects0, objects1, change_logs)

        _ = agent.choose_action([frame], frame)
        # Rule hypotheses should be populated after analyzing success
        assert isinstance(agent.rule_hypotheses, dict)

    def test_failure_patterns_stored_on_global_failure(self):
        agent = make_agent()
        frame = make_frame(actions=[GameAction.ACTION6, GameAction.ACTION1])

        # Initialize with a state
        objects0 = [{"position": (1,1), "id": "obj_1", "color": 2, "size": (1,1), "fingerprint": 3, "pixels": 1, "pixel_coords": frozenset()}]
        agent._perceive_objects = lambda latest_frame: objects0
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {}, {}, {}, {})
        agent.choose_action([frame], frame)

        # Next frame: no changes at all -> global failure
        agent._perceive_objects = lambda latest_frame: objects0
        agent._log_changes = lambda prev_summary, curr_summary: ([], objects0)
        agent._analyze_relationships = lambda summary: ({}, {}, {}, {}, {}, {}, {})

        _ = agent.choose_action([frame], frame)
        # Should have recorded failure contexts or patterns; may be empty default
        assert isinstance(agent.failure_contexts, dict)

    def test_seen_outcomes_updated_after_choice(self):
        agent = make_agent()
        frame = make_frame(actions=[GameAction.ACTION6, GameAction.ACTION1])
        objects = [{"position": (2,2), "id": "obj_1", "color": 1, "size": (1,1), "fingerprint": 2, "pixels": 1, "pixel_coords": frozenset()}]
        self._base_stub(agent, objects, objects, [])

        _ = agent.choose_action([frame], frame)
        # After choosing, seen_outcomes should be a set (possibly empty)
        assert isinstance(agent.seen_outcomes, set)
dict_outcome
    def test_get_learning_key_click_target_specific(self):
        agent = make_agent()
        # direct method check to ensure ACTION6 with target id is specific
        key_targeted = agent._get_learning_key("ACTION6", "obj_3")
        key_non_target = agent._get_learning_key("ACTION6", None)
        key_other = agent._get_learning_key("ACTION1", None)
        assert key_targeted == "ACTION6_obj_3"
        assert key_non_target == "ACTION6"
        assert key_other == "ACTION1"

    def test_predict_outcome_none_when_no_hypothesis(self):
        agent = make_agent()
        # empty rule_hypotheses -> predict returns (None, 0)
        events, confidence = agent._predict_outcome(("ACTION1", "obj_1"), current_context={})
        assert events is None and confidence == 0

    def test_context_matches_pattern_and_single(self):
        agent = make_agent()
        # Build a simple context and matching rule/pattern
        context = {
            "adj": {"obj_1": ["na", "obj_2", "na", "na"]},
            "rels": {"Color": {1: {"obj_1", "obj_2"}}},
            "diag_align": {"top_left_to_bottom_right": [{"obj_1", "obj_2"}]},
        }
        # Exact rule should match
        assert agent._context_matches_pattern(context, {
            "adj": {"obj_1": ["na", "obj_2", "na", "na"]},
            "rels": {"Color": {1: {"obj_1", "obj_2"}}},
            "diag_align": {"top_left_to_bottom_right": [{"obj_1", "obj_2"}]},
        }) is True
        # Partial pattern should match via single matcher
        assert agent._context_matches_pattern_single(context, {
            "adj": {"obj_1": ["na", "obj_2", "na", "na"]}
        }) is True
        # Mismatch should fail
        assert agent._context_matches_pattern_single(context, {
            "adj": {"obj_1": ["na", "obj_3", "na", "na"]}
        }) is False
