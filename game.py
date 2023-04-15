import numpy as np
import random
from MCTS import MCTSTreeNode
from copy import deepcopy
from recursive_bayes_filter import ParticleFilter


class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52])
        self.N_BLOCKS_PER = len(self.state)//2 - 1
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        TODO: You need to implement this.
        """
        col, row = cr
        return col+row*self.N_COLS

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        return (n % self.N_COLS, n//self.N_COLS)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a terminal board state, and return True or False.

        TODO: You need to implement this.
        """
        if not self.is_valid():
            return False

        white_ball_dec = self.decode_state[self.N_BLOCKS_PER]
        black_ball_dec = self.decode_state[-1]
        # ADD a condition for stalemate or no moves possible, ie, draw
        return white_ball_dec[1] == self.N_ROWS-1 or black_ball_dec[1] == 0

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)

        TODO: You need to implement this.
        """
        if not issubclass(self.state.dtype.type, np.integer):
            return False
        elif np.min(self.state) < 0 or np.max(self.state) > self.encode_single_pos((self.N_COLS-1, self.N_ROWS-1)):
            return False

        # check structure is correct or not
        elif len(np.unique(self.state)) != len(self.state) - 2:  # ensures only 2 repetitions
            return False
        # this ensures those 2 repetitions are of only the balls and also that all blocks and all balls dont overlap each other in any restricted way
        elif not (self.state[self.N_BLOCKS_PER] in self.state[:self.N_BLOCKS_PER] and self.state[-1] in self.state[self.N_BLOCKS_PER+1:-1]):
            return False
        # both cannot win simultaneously
        elif self.decode_state[self.N_BLOCKS_PER][1] == self.N_ROWS-1 and self.decode_state[-1][1] == 0:
            return False
        else:
            return True


class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.

        TODO: You need to implement this.
        """
        def check_validity_decoded_pos_and_encode(colf, rowf):
            n_rows = board_state.N_ROWS
            n_cols = board_state.N_COLS
            enc = board_state.encode_single_pos((colf, rowf))
            if colf < 0 or colf >= n_cols or rowf < 0 or rowf >= n_rows:
                return None
            elif enc in board_state.state:
                return None
            else:
                return enc

        actions = set()
        enc = board_state.state[piece_idx]

        # Can't move if has the ball
        if enc == board_state.state[board_state.N_BLOCKS_PER] or enc == board_state.state[-1]:
            return actions
        col, row = board_state.decode_state[piece_idx]

        actions.add(check_validity_decoded_pos_and_encode(col+1, row+2))
        actions.add(check_validity_decoded_pos_and_encode(col+1, row-2))
        actions.add(check_validity_decoded_pos_and_encode(col-1, row+2))
        actions.add(check_validity_decoded_pos_and_encode(col-1, row-2))
        actions.add(check_validity_decoded_pos_and_encode(col+2, row+1))
        actions.add(check_validity_decoded_pos_and_encode(col-2, row+1))
        actions.add(check_validity_decoded_pos_and_encode(col+2, row-1))
        actions.add(check_validity_decoded_pos_and_encode(col-2, row-1))
        actions.discard(None)
        return actions

    @staticmethod
    def validate_onestep_ball_action_(board_state, player_idx, col_, row_, c, r, opp_blocks=True):
        if opp_blocks:
            opp_blocks_decs = board_state.decode_state[(board_state.N_BLOCKS_PER+1) * (1-player_idx):(board_state.N_BLOCKS_PER+1) * (1-player_idx)+board_state.N_BLOCKS_PER]
        else:
            opp_blocks_decs = []

        def has_horiz_path():
            if row_ != r:
                return False

            for o in opp_blocks_decs:
                opc, opr = o
                if opr == row_ and (col_ < opc < c or c < opc < col_):
                    return False
            return True

        def has_vertical_path():
            if col_ != c:
                return False

            for o in opp_blocks_decs:
                opc, opr = o
                if opc == col_ and (row_ < opr < r or r < opr < row_):
                    return False
            return True

        def has_diag_path():
            if abs(col_ - c) != abs(row_ - r):
                return False

            for o in opp_blocks_decs:
                opc, opr = o
                if abs(col_ - opc) == abs(row_ - opr) and ((col_ < opc < c and row_ < opr < r) or (col_ < opc < c and r < opr < row_) or (c < opc < col_ and r < opr < row_) or (c < opc < col_ and row_ < opr < r)):
                    return False
            return True

        return has_horiz_path() or has_vertical_path() or has_diag_path()

    @staticmethod
    def single_ball_actions(board_state, player_idx, validate_action=None, opp_blocks=True):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.

        TODO: You need to implement this.
        """
        actions = set()
        explore = set()
        unexplored = set()

        enc = board_state.state[(board_state.N_BLOCKS_PER+1) * player_idx + board_state.N_BLOCKS_PER]
        col, row = board_state.decode_single_pos(enc)
        blocks_decs = board_state.decode_state[(board_state.N_BLOCKS_PER+1) * player_idx:(board_state.N_BLOCKS_PER+1) * player_idx+board_state.N_BLOCKS_PER]

        explore.add((col, row))
        unexplored.update(blocks_decs)

        while True:
            if len(explore) == 0:
                break
            col_, row_ = explore.pop()
            for u in unexplored:
                c, r = u
                if Rules.validate_onestep_ball_action_(board_state, player_idx, col_, row_, c, r, opp_blocks):
                    if validate_action is not None and validate_action == board_state.encode_single_pos(u):
                        # Found the action
                        return True
                    actions.add(u)
                    explore.add(u)
            unexplored = unexplored - actions
        actions.discard((col, row))
        actions_enc = set()
        for act in actions:
            actions_enc.add(board_state.encode_single_pos(act))

        if validate_action is not None:
            return False

        return actions_enc


class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players, gui=None, tries_per_round=7):
        self.game_state = BoardState()
        # The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.current_round = -1
        self.players = players
        self.gui = gui
        self.max_tries_per_round = tries_per_round
        self.BASE_OBS_MODEL = [0.6, 0.1, 0.1, 0.1, 0.1]  # self, top, right, bottom, left

    def write_to_file(self, pid, NUM):
        def get_obs_dstate():
            observation0 = self.sample_observation(0)
            observation1 = self.sample_observation(1)
            observation = deepcopy(observation0)
            observation[self.game_state.N_BLOCKS_PER+1:] = observation1[self.game_state.N_BLOCKS_PER+1:]
            return tuple((tuple(observation), pid))

        ground_truth_filepath = f"data/ground_truth_sequences/{NUM}.txt"
        obs_filepath = f"data/observed_sequences/{NUM}.txt"
        ground_truth_dstate = tuple((tuple(self.game_state.make_state()), pid))
        obs_dstate = get_obs_dstate()
        with open(ground_truth_filepath, "a") as f:
            f.write(f"{ground_truth_dstate}\n")
        with open(obs_filepath, "a") as f:
            f.write(f"{obs_dstate}\n")

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            # Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            # For the player who needs to move, provide them with the current game state
            # and then ask them to choose an action according to their policy
            # self.write_to_file(player_idx, NUM)
            observation = self.sample_observation((player_idx + 1) % 2)
            is_valid_action = False
            tries = 0
            while (not is_valid_action) and (tries < self.max_tries_per_round):
                action, value = self.players[player_idx].policy(observation)
                try:
                    is_valid_action = self.validate_action(action, player_idx)
                except ValueError:
                    is_valid_action = False
                tries += 1
                print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value} Validity: {is_valid_action}")
                self.players[player_idx].process_feedback(observation, action, is_valid_action)

            if not is_valid_action:
                # If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black provided an invalid action"

            # Updates the game state
            self.update(action, player_idx)

            if self.gui is not None:
                self.gui.highlight_move((self.game_state.N_BLOCKS_PER+1)*player_idx + action[0])
                self.gui.set_board_state(list(self.game_state.state))
                self.gui.update_state()

        # self.write_to_file((player_idx+1) % 2, NUM)

        # Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def get_abs_from_rel_index(self, rel_i, player_i):
        return (self.game_state.N_BLOCKS_PER+1)*player_i + rel_i

    def generate_valid_actions(self, player_idx: int, only_pieces=None):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.

        TODO: You need to implement this.
        """
        all_actions = set()
        # For boxes
        for rel_i in range(self.game_state.N_BLOCKS_PER):
            if only_pieces is not None and rel_i not in only_pieces:
                continue
            abs_i = self.get_abs_from_rel_index(rel_i, player_idx)
            block_actions = Rules.single_piece_actions(self.game_state, abs_i)
            for b in block_actions:
                all_actions.add((rel_i, b))

        # For the ball (rel_i = 5)
        if only_pieces is not None and self.game_state.N_BLOCKS_PER not in only_pieces:
            return all_actions
        ball_actions = Rules.single_ball_actions(self.game_state, player_idx)
        for b in ball_actions:
            all_actions.add((self.game_state.N_BLOCKS_PER, b))

        return all_actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError

        TODO: You need to implement this.
        """
        def queen_path_exists_no_opp():
            return Rules.single_ball_actions(self.game_state, player_idx, validate_action=action[1], opp_blocks=False)

        def queen_path_exists_opp():
            return Rules.single_ball_actions(self.game_state, player_idx, validate_action=action[1], opp_blocks=True)

        def raise_error():
            rid, _ = action
            if rid == self.game_state.N_BLOCKS_PER:
                raise_error_ball()
            else:
                raise_error_block()

        def raise_error_ball():
            """
            Exit Codes:
                -2: Going out of board
                -1: your piece not at destination
                0: No multiple-moves-allowed queen path exists
                1: has a path but opp piece in the way
            """
            if encf < 0 or encf > self.game_state.encode_single_pos((self.game_state.N_COLS-1, self.game_state.N_ROWS-1)):
                raise ValueError(
                    "Action Invalid for the Ball: Going out of Board!")
            elif encf not in blocks_encs:
                raise ValueError(
                    "Action Invalid for the Ball: None of your blocks is at the destination.")
            elif not queen_path_exists_no_opp():
                raise ValueError(
                    "Action Invalid for the Ball: A valid path with queen moves does not exist.")
            elif not queen_path_exists_opp():
                raise ValueError(
                    "Action Invalid for the Ball: A path exists but opponent piece intersects the path.")
            else:
                raise ValueError(
                    "Action Invalid for the Ball: ****DEBUGGING REQUIRED**** to find out source of invalidity.")

        def raise_error_block():
            """
            Exit Codes:
                -1: Not a knight move
                0: Ball on it
                1: going out of board
                2: your own piece exists at destination
                3: opp piece exists at destination
            """
            if encf < 0 or encf > self.game_state.encode_single_pos((self.game_state.N_COLS-1, self.game_state.N_ROWS-1)):
                raise ValueError(
                    "Action Invalid for the Block: Going out of Board!")
            elif not ((abs(dec[0] - decf[0]) == 1 and abs(dec[1] - decf[1]) == 2) or (abs(dec[0] - decf[0]) == 2 and abs(dec[1] - decf[1]) == 1)):
                raise ValueError(
                    "Action Invalid for the Block: Not a knight move!")
            elif enc == self.game_state.state[self.get_abs_from_rel_index(self.game_state.N_BLOCKS_PER, player_idx)]:
                raise ValueError(
                    "Action Invalid for the Block: Block cannot be moved with the ball on it.")
            elif encf in blocks_encs:
                raise ValueError(
                    "Action Invalid for the Block: One of your blocks is already at the destination.")
            elif encf in opp_blocks_encs:
                raise ValueError(
                    "Action Invalid for the Block: One of your opponent's blocks is already at the destination.")
            else:
                raise ValueError(
                    "Action Invalid for the Block: ****DEBUGGING REQUIRED**** to find out source of invalidity.")

        rid, encf = action
        id = self.get_abs_from_rel_index(rid, player_idx)
        enc = self.game_state.state[id]
        dec = self.game_state.decode_state[id]
        decf = self.game_state.decode_single_pos(encf)
        blocks_encs = self.game_state.state[(self.game_state.N_BLOCKS_PER+1) * player_idx:(
            self.game_state.N_BLOCKS_PER+1) * player_idx+self.game_state.N_BLOCKS_PER]
        opp_blocks_encs = self.game_state.state[(self.game_state.N_BLOCKS_PER+1) * (1-player_idx):(
            self.game_state.N_BLOCKS_PER+1) * (1-player_idx)+self.game_state.N_BLOCKS_PER]

        if action in self.generate_valid_actions(player_idx):
            return True
        else:
            raise_error()

    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * (self.game_state.N_BLOCKS_PER+1)  # Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)

    def modify_observation_model(self, abs_idx, ground_truth_decode_state):
        def is_valid_decode(col, row):
            n_rows = self.game_state.N_ROWS
            n_cols = self.game_state.N_COLS
            if col < 0 or col >= n_cols or row < 0 or row >= n_rows:
                return False
            return True
        modified_obs_model = deepcopy(self.BASE_OBS_MODEL)
        pos = ground_truth_decode_state[abs_idx]
        pos_top = (pos[0], pos[1] + 1)
        pos_bottom = (pos[0], pos[1] - 1)
        pos_left = (pos[0] - 1, pos[1])
        pos_right = (pos[0] + 1, pos[1])
        observed_pos = [pos, pos_top, pos_right, pos_bottom, pos_left]

        if not is_valid_decode(*pos_top) or pos_top in ground_truth_decode_state:
            modified_obs_model[0] += modified_obs_model[1]
            modified_obs_model[1] = 0
        if not is_valid_decode(*pos_right) or pos_right in ground_truth_decode_state:
            modified_obs_model[0] += modified_obs_model[2]
            modified_obs_model[2] = 0
        if not is_valid_decode(*pos_bottom) or pos_bottom in ground_truth_decode_state:
            modified_obs_model[0] += modified_obs_model[3]
            modified_obs_model[3] = 0
        if not is_valid_decode(*pos_left) or pos_left in ground_truth_decode_state:
            modified_obs_model[0] += modified_obs_model[4]
            modified_obs_model[4] = 0
        return modified_obs_model, observed_pos

    def sample_observation(self, opposing_idx):
        """
        Returns a sample board state, according to the observation model. The format of the
        return value is the same as in BoardState.make_state() -- returns a list of tuples.
        TODO: You need to implement the observation model. As currently implemented, this operates
        exactly as it did in Assignment 3.
        """
        ground_truth_state = self.game_state.make_state()
        sampled_obs = deepcopy(ground_truth_state)
        opp_ball_idx = self.get_abs_from_rel_index(self.game_state.N_BLOCKS_PER, opposing_idx)
        for rel_idx in range(self.game_state.N_BLOCKS_PER):
            abs_idx = self.get_abs_from_rel_index(rel_idx, opposing_idx)
            obs_model, observed_pos = self.modify_observation_model(abs_idx, ground_truth_state)
            sampled_obs[abs_idx] = random.choices(observed_pos, obs_model, k=1)[0]
            if ground_truth_state[abs_idx] == ground_truth_state[opp_ball_idx]:
                sampled_obs[opp_ball_idx] = sampled_obs[abs_idx]
        return sampled_obs


class Player:
    def __init__(self, policy_fnc):
        self.policy_fnc = policy_fnc

    def policy(self, decode_state):
        pass

    def process_feedback(self, observation, action, is_valid):
        pass


class AdversarialSearchPlayer(Player):
    def __init__(self, gsp, player_idx):
        """
        You can customize the signature of the constructor above to suit your needs.
        In this example, in the above parameters, gsp is a GameStateProblem, and
        gsp.adversarial_search_method is a method of that class.
        """
        super().__init__(gsp.adversarial_search_method)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx

    def policy(self, decode_state):
        """
        Here, the policy of the player is to consider the current decoded game state
        and then correctly encode it and provide any additional required parameters to the
        assigned policy_fnc (which in this case is gsp.adversarial_search_method), and then
        return the result of self.policy_fnc
        """
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        val_a, val_b, val_c = (1, 2, 3)
        return self.policy_fnc(state_tup, val_a, val_b, val_c)


class VanillaMCTSPlayer(Player):
    def __init__(self, gsp, player_idx, playout, iters=1000, cycle=True, rollout_heu=True, selection_heu=True, opponent_policy="mcts", EPS=0.1, ALPHA=4, BETA=0.75):
        super().__init__(gsp.vanilla_mcts)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.playout = playout
        self.iters = iters
        self.cycle = cycle
        self.rollout_heu = rollout_heu
        self.selection_heu = selection_heu
        self.opponent_policy = opponent_policy
        self.EPS = EPS
        self.ALPHA = ALPHA
        self.BETA = BETA

    def policy(self, decode_state):
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        return self.policy_fnc(state_tup, self.playout, iters=self.iters, cycle=self.cycle, rollout_heu=self.rollout_heu, selection_heu=self.selection_heu, opponent_policy=self.opponent_policy, EPS=self.EPS, ALPHA=self.ALPHA, BETA=self.BETA)


class ProbabilisticVanillaMCTSPlayer(Player):
    def __init__(self, gsp, player_idx, playout, ini_dstate, method="dirac", iters=1000, cycle=True, rollout_heu=True, selection_heu=True, opponent_policy="mcts", EPS=0.1, ALPHA=4, BETA=0.75):
        super().__init__(gsp.vanilla_mcts)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.playout = playout
        self.iters = iters
        self.cycle = cycle
        self.rollout_heu = rollout_heu
        self.selection_heu = selection_heu
        self.opponent_policy = opponent_policy
        self.EPS = EPS
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.retry = False
        self.particle_filter = ParticleFilter(self.player_idx, ini_dstate, deepcopy(self.gsp.sim), method=method)

    def process_feedback(self, observation, action, is_valid):
        if is_valid:
            self.particle_filter.belief_propagate_own_action(action, observation)
        else:
            self.particle_filter.process_feedback_invalid_action(action, observation)
        self.retry = (not is_valid)

    def get_MLE_decode_state(self, observation):
        if not self.retry:
            self.particle_filter.belief_update_observation(observation)
        mle = self.particle_filter.get_belief_MLE()
        print(f"#################### Player {self.player_idx} has state confidence: {mle[1]}, amongst num_particles: {mle[2]}")
        return mle[0][0]

    def policy(self, observation):
        # Here use the player's particle filter and get best estimate of current state
        decode_state = self.get_MLE_decode_state(observation)
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        return self.policy_fnc(state_tup, self.playout, iters=self.iters, cycle=self.cycle, rollout_heu=self.rollout_heu, selection_heu=self.selection_heu, opponent_policy=self.opponent_policy, EPS=self.EPS, ALPHA=self.ALPHA, BETA=self.BETA)


class ProbabilisticRandomPlayer(Player):
    def __init__(self, gsp, player_idx, playout, ini_dstate, method="dirac"):
        super().__init__(None)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.playout = playout
        self.retry = False
        self.particle_filter = ParticleFilter(self.player_idx, ini_dstate, deepcopy(self.gsp.sim), method=method)

    def process_feedback(self, observation, action, is_valid):
        if is_valid:
            self.particle_filter.belief_propagate_own_action(action, observation)
        else:
            self.particle_filter.process_feedback_invalid_action(action, observation)
        self.retry = (not is_valid)

    def get_MLE_decode_state(self, observation):
        if not self.retry:
            self.particle_filter.belief_update_observation(observation)
        mle = self.particle_filter.get_belief_MLE()
        print(f"#################### Player {self.player_idx} has state confidence: {mle[1]}, amongst num_particles: {mle[2]}")
        return mle[0][0]

    def policy(self, observation):
        decode_state = self.get_MLE_decode_state(observation)
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        root = MCTSTreeNode(state_tup, self.gsp.sim, self.gsp.get_actions(state_tup))
        if len(self.playout.game) > 0:
            for child in self.playout.game[-1].children:
                if child.state == state_tup:
                    root = child
                    # root.parent = None  # not required actually, let it go through upto the original root (because thats required in MCTSPlayRecordedGame)
                    break
        self.playout.add_node(root)
        return random.choice(list(self.gsp.get_actions(state_tup))), 0


class RandomPlayer(Player):
    def __init__(self, gsp, player_idx, playout):
        super().__init__(None)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.playout = playout

    def policy(self, decode_state):
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        root = MCTSTreeNode(state_tup, self.gsp.sim, self.gsp.get_actions(state_tup))
        if len(self.playout.game) > 0:
            for child in self.playout.game[-1].children:
                if child.state == state_tup:
                    root = child
                    # root.parent = None  # not required actually, let it go through upto the original root (because thats required in MCTSPlayRecordedGame)
                    break
        self.playout.add_node(root)
        return random.choice(list(self.gsp.get_actions(state_tup))), 0


class PassivePlayer(Player):
    def __init__(self, gsp, player_idx, playout):
        super().__init__(None)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.playout = playout

    def policy(self, decode_state):
        encoded_state_tup = tuple(self.b.encode_single_pos(s) for s in decode_state)
        state_tup = tuple((encoded_state_tup, self.player_idx))
        root = MCTSTreeNode(state_tup, self.gsp.sim, self.gsp.get_actions(state_tup))
        if len(self.playout.game) > 0:
            for child in self.playout.game[-1].children:
                if child.state == state_tup:
                    root = child
                    # root.parent = None  # not required actually, let it go through upto the original root (because thats required in MCTSPlayRecordedGame)
                    break
        self.playout.add_node(root)
        self.b.state = np.array(encoded_state_tup)
        self.b.decode_state = self.b.make_state()
        ball_actions = Rules.single_ball_actions(self.b, self.player_idx)
        actions = [(self.gsp.sim.game_state.N_BLOCKS_PER, e) for e in ball_actions]
        return random.choice(actions), 0
