import numpy as np
import queue
import pytest
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem
from report import *


class TestSearch:

    def test_game_state_goal_state(self):
        b1 = BoardState()
        gsp = GameStateProblem(b1, b1, 0)

        sln = gsp.search_alg_fnc()
        ref = [(tuple((tuple(b1.state), 0)), None)]

        assert sln == ref

    # NOTE: If you'd like to test multiple variants of your algorithms, enter their keys below
    # in the parametrize function. Your set_search_alg should then set the correct method to
    # use.

    @pytest.mark.parametrize("alg", ["BFS"])
    def test_game_state_problem(self, alg):
        """
        Tests search based planning
        """
        b1 = BoardState()
        b2 = BoardState()
        b2.update(0, 14)

        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        # Single Step
        ref = [(tuple((tuple(b1.state), 0)), (0, 14)), (tuple((tuple(b2.state), 1)), None)]
        assert sln == ref

        b2 = BoardState()
        b2.update(0, 23)

        gsp = GameStateProblem(b1, b2, 0)
        gsp.set_search_alg(alg)
        sln = gsp.search_alg_fnc()

        # Two Step:
        # (0, 14) or (0, 10) -> (any) -> (0, 23) -> (undo any) -> (None, goal state)

        # print(gsp.goal_state_set)
        # print(sln)
        assert len(sln) == 5  # Player 1 needs to move once, then move the piece back
        assert sln[0] == (tuple((tuple(b1.state), 0)), (0, 14)) or sln[0] == (tuple((tuple(b1.state), 0)), (0, 10))
        assert sln[1][0][1] == 1
        assert sln[2][1] == (0, 23)
        assert sln[4] == (tuple((tuple(b2.state), 0)), None)

    def test_initial_state(self):
        """
        Confirms the initial state of the game board
        """
        board = BoardState()
        assert board.decode_state == board.make_state()

        ref_state = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (3, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (3, 7)]

        assert board.decode_state == ref_state

    def test_generate_actions(self):
        sim = GameSimulator(None)
        generated_actions = sim.generate_valid_actions(0)
        assert (0, 6) not in generated_actions
        assert (4, 0) not in generated_actions

    # NOTE: You are highly encouraged to add failing test cases here
    # in order to test your validate_action implementation. To add an
    # invalid action, fill in the action tuple, the player_idx, the
    # validity boolean (would be False for invalid actions), and a
    # unique portion of the descriptive error message that your raised
    # ValueError should return. For example, if you raised:
    # ValueError("Cannot divide by zero"), then you would pass some substring
    # of that description for val_msg.
    @pytest.mark.parametrize("action,player,is_valid,val_msg", [
        ((0, 15), 0, False, "knight move"),
        ((1, 160), 0, False, "out of"),
        ((2, 10), 1, False, "knight move"),
        ((5, 6), 0, False, "None of your blocks"),
        ((5, 3), 1, False, "None of your blocks"),
    ])
    def test_validate_action(self, action, player, is_valid, val_msg):
        sim = GameSimulator(None)
        if is_valid:
            assert sim.validate_action(action, player) == is_valid
        else:
            with pytest.raises(ValueError) as exinfo:
                result = sim.validate_action(action, player)
            assert val_msg in str(exinfo.value)

    @pytest.mark.parametrize("state,is_term", [
        ([1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52], False),  # Initial State
        ([1, 2, 3, 4, 5, 55, 50, 51, 52, 53, 54, 0], False),  # Invalid State
        ([1, 2, 3, 4, 49, 49, 50, 51, 52, 53, 54, 0], False),  # Invalid State
        ([1, 2, 3, 4, 49, 49, 50, 51, 52, 53, 54, 54], True),  # Player 1 wins
        ([1, 2, 3, 4, 5, 5, 50, 51, 52, 53, 6, 6], True),  # Player 2 wins
        ([1, 2, 3, 4, 5, 5, 50, 4, 52, 53, 6, 6], False),  # Invalid State
    ])
    def test_termination_state(self, state, is_term):
        board = BoardState()
        board.state = np.array(state)
        board.decode_state = board.make_state()

        assert board.is_termination_state() == is_term

    def test_encoded_decode(self):
        board = BoardState()
        assert board.decode_state == [board.decode_single_pos(x) for x in board.state]

        enc = np.array([board.encode_single_pos(x) for x in board.decode_state])
        assert np.all(enc == board.state)

    def test_is_valid(self):
        board = BoardState()
        assert board.is_valid()

        # Out of bounds test
        board.update(0, -1)
        assert not board.is_valid()

        board.update(0, 0)
        assert board.is_valid()

        # Out of bounds test
        board.update(0, -1)
        board.update(6, 56)
        assert not board.is_valid()

        # Overlap test
        board.update(0, 0)
        board.update(6, 0)
        assert not board.is_valid()

        # Ball is on index 0
        board.update(5, 1)
        board.update(0, 1)
        board.update(6, 50)
        assert board.is_valid()

        # Player is not holding the ball
        board.update(5, 0)
        assert not board.is_valid()

        board.update(5, 10)
        assert not board.is_valid()

    @pytest.mark.parametrize("state,reachable,player", [
        (
            [
                (1, 1), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1),
                (0, 0), (2, 0), (0, 2), (2, 2), (3, 3), (3, 3)
            ],
            set([(0, 1), (2, 1), (1, 2), (1, 0)]),
            0
        ),
        (
            [
                (1, 1), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1),
                (0, 0), (2, 0), (0, 2), (2, 2), (3, 3), (3, 3)
            ],
            set([(2, 2)]),
            1
        ),
        (
            [
                (1, 1), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1),
                (0, 0), (2, 0), (0, 2), (2, 2), (3, 3), (0, 0)
            ],
            set(),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(2, 0), (0, 2), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(0, 0), (0, 2), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 2),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(0, 0), (2, 0), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 2),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(0, 0), (2, 0), (0, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 3),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(0, 0), (2, 0), (0, 2), (2, 2)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (0, 1)
            ],
            set([(2, 1), (3, 1), (3, 2), (2, 3)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (2, 1)
            ],
            set([(0, 1), (3, 1), (3, 2), (2, 3)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (3, 1)
            ],
            set([(0, 1), (2, 1), (3, 2), (2, 3)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (3, 2)
            ],
            set([(0, 1), (2, 1), (3, 1), (2, 3)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (2, 3), (2, 3)
            ],
            set([(0, 1), (2, 1), (3, 1), (3, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (1, 2)
            ],
            set([(0, 1), (2, 1), (3, 1), (3, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (0, 1)
            ],
            set([(2, 1), (3, 1), (3, 2), (1, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (2, 1)
            ],
            set([(0, 1), (3, 1), (3, 2), (1, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 1)
            ],
            set([(0, 1), (2, 1), (3, 2), (1, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(0, 1), (2, 1), (3, 1), (1, 2)]),
            1
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(2, 0), (0, 2), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 0),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(0, 0), (0, 2), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 2),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(0, 0), (2, 0), (2, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 2),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(0, 0), (2, 0), (0, 2), (0, 3)]),
            0
        ),
        (
            [
                (0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (0, 3),
                (0, 1), (2, 1), (3, 1), (3, 2), (1, 2), (3, 2)
            ],
            set([(0, 0), (2, 0), (0, 2), (2, 2)]),
            0
        ),
    ])
    def test_ball_reachability(self, state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        assert predicted_reachable_encoded == encoded_reachable

    @pytest.mark.parametrize("NUM,method", [
        (1, "dirac"),
        (2, "dirac"),
        (3, "dirac"),
        (4, "dirac"),
        (5, "dirac"),
        (6, "dirac"),
        (7, "dirac"),
        (8, "dirac"),
        (9, "dirac"),
        (10, "dirac"),
        (11, "dirac"),
        (12, "dirac"),
        (13, "dirac"),
        (14, "dirac"),
        (15, "dirac"),
        (16, "dirac"),
        (17, "dirac"),
        (18, "dirac"),
        (19, "dirac"),
        (20, "dirac"),
        (21, "dirac"),
        (22, "dirac"),
        (23, "dirac"),
        (24, "dirac"),
        (25, "dirac"),
        (26, "dirac"),
        (27, "dirac"),
        (28, "dirac"),
        (29, "dirac"),
        (30, "dirac"),
        (31, "dirac"),
        (32, "dirac"),
        (33, "dirac"),
        (34, "dirac"),
        (35, "dirac"),
        (36, "dirac"),
        (37, "dirac"),
        (38, "dirac"),
        (39, "dirac"),
        (40, "dirac"),
        (1, "random"),
        (2, "random"),
        (3, "random"),
        (4, "random"),
        (5, "random"),
        (6, "random"),
        (7, "random"),
        (8, "random"),
        (9, "random"),
        (10, "random"),
        (11, "random"),
        (12, "random"),
        (13, "random"),
        (14, "random"),
        (15, "random"),
        (16, "random"),
        (17, "random"),
        (18, "random"),
        (19, "random"),
        (20, "random"),
        (21, "random"),
        (22, "random"),
        (23, "random"),
        (24, "random"),
        (25, "random"),
        (26, "random"),
        (27, "random"),
        (28, "random"),
        (29, "random"),
        (30, "random"),
        (31, "random"),
        (32, "random"),
        (33, "random"),
        (34, "random"),
        (35, "random"),
        (36, "random"),
        (37, "random"),
        (38, "random"),
        (39, "random"),
        (40, "random"),
    ])
    def test_infer_last_state_from_seq(self, NUM, method):
        b1 = BoardState()
        ini_dstate = tuple((tuple(b1.decode_state), 0))
        q1 = Q1(NUM, ini_dstate, GameSimulator(None), method=method)
        for i, obs_dstate in enumerate(q1.obs_seq):
            q1.belief_update_observation(*obs_dstate)
        weight = q1.weighted_particles[q1.ground_truth_seq[-1]]
        assert weight >= 0.5

    @pytest.mark.parametrize("NUM,method", [
        (1, "dirac"),
        (2, "dirac"),
        (3, "dirac"),
        (4, "dirac"),
        (5, "dirac"),
        (6, "dirac"),
        (7, "dirac"),
        (8, "dirac"),
        (9, "dirac"),
        (10, "dirac"),
        (11, "dirac"),
        (12, "dirac"),
        (13, "dirac"),
        (14, "dirac"),
        (15, "dirac"),
        (16, "dirac"),
        (17, "dirac"),
        (18, "dirac"),
        (19, "dirac"),
        (20, "dirac"),
        (21, "dirac"),
        (22, "dirac"),
        (23, "dirac"),
        (24, "dirac"),
        (25, "dirac"),
        (26, "dirac"),
        (27, "dirac"),
        (28, "dirac"),
        (29, "dirac"),
        (30, "dirac"),
        (31, "dirac"),
        (32, "dirac"),
        (33, "dirac"),
        (34, "dirac"),
        (35, "dirac"),
        (36, "dirac"),
        (37, "dirac"),
        (38, "dirac"),
        (39, "dirac"),
        (40, "dirac"),
        (1, "random"),
        (2, "random"),
        (3, "random"),
        (4, "random"),
        (5, "random"),
        (6, "random"),
        (7, "random"),
        (8, "random"),
        (9, "random"),
        (10, "random"),
        (11, "random"),
        (12, "random"),
        (13, "random"),
        (14, "random"),
        (15, "random"),
        (16, "random"),
        (17, "random"),
        (18, "random"),
        (19, "random"),
        (20, "random"),
        (21, "random"),
        (22, "random"),
        (23, "random"),
        (24, "random"),
        (25, "random"),
        (26, "random"),
        (27, "random"),
        (28, "random"),
        (29, "random"),
        (30, "random"),
        (31, "random"),
        (32, "random"),
        (33, "random"),
        (34, "random"),
        (35, "random"),
        (36, "random"),
        (37, "random"),
        (38, "random"),
        (39, "random"),
        (40, "random"),
    ])
    def test_infer_actions_from_seq(self, NUM, method):
        b1 = BoardState()
        ini_dstate = tuple((tuple(b1.decode_state), 0))
        q1 = Q1(NUM, ini_dstate, GameSimulator(None), method=method)
        TOTAL_ACTIONS = len(q1.ground_truth_seq) - 1
        MATCHED_ACTIONS = 0
        for i, obs_dstate in enumerate(q1.obs_seq):
            q1.belief_update_observation(*obs_dstate)
            if i > 0:
                if q1.ground_truth_act_seq[i-1] in q1.weighted_actions.keys():
                    if q1.weighted_actions[q1.ground_truth_act_seq[i-1]] >= 0.5:
                        MATCHED_ACTIONS += 1
        assert MATCHED_ACTIONS / TOTAL_ACTIONS >= 0.85
