import numpy as np
import math
import random
from copy import deepcopy
import time
from board_gui import EOTgui
from matplotlib import pyplot as plt


class MCTSPlayout:
    def __init__(self):
        self.game = []

    def add_node(self, node):
        self.game.append(node)


class MCTSPlayRecordedGame:
    def __init__(self, playout, initial_state, tsec=2):
        self.game = playout.game
        self.initial_state = initial_state
        self.tsec = tsec
        self.gui = EOTgui(tsec=self.tsec, ini=self.initial_state)

    def play(self):
        self.gui.initialize_pieces()
        last_mover_recorded_idx = self.game[-1].state[1]
        WINNER = "WHITE" if last_mover_recorded_idx == 0 else "BLACK"
        print(f"#### Starting game: WINNER is {WINNER} ####")
        for node in self.game[1:]:
            print(f"Visited the upcoming state {node.num_visits} times")
            self.gui.highlight_move((1+self.gui.N_BLOCKS_PER)*node.parent.state[1] + node.parent_action[0])
            self.gui.set_board_state(list(node.state[0]))
            self.gui.update_state()
        print(f"#### End of game: WINNER is {WINNER} ####")
        plt.show()


class MCTSTreeNode:
    def __init__(self, state, sim, untried_actions, parent=None, parent_action=None, cycle=False, rollout_heu=False, selection_heu=False, EPS=0.1, ALPHA=4, BETA=0.75, mcts_idx=0, opponent_policy="mcts"):
        self.state = state
        self.sim = sim
        self.untried_actions = untried_actions
        self.parent = parent
        self.parent_action = parent_action
        self.cycle = cycle
        self.rollout_heu = rollout_heu
        self.selection_heu = selection_heu
        self.EPS = EPS
        self.children = []
        self.num_visits = 0
        self.wins = {0: 0, 1: 0}
        self.shortest_win_seen = {0: math.inf, 1: math.inf}
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.mcts_idx = mcts_idx
        self.opponent_policy = opponent_policy

    @staticmethod
    def _UCB1formula(w, n, N, BETA, c=math.sqrt(2)):
        return BETA*(w/n) + (2-BETA) * c * math.sqrt(math.log(N)/n)

    def _update_sim_state(self, state):
        self.sim.game_state.state = np.array(state[0])
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

    def _my_sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _selection_heuristic(self, child, pidx, sim_temp, ALPHA):
        # Calculates the inverse of distance to end for the child
        sim_copy_temp = deepcopy(sim_temp)
        sim_copy_temp.game_state.state = np.array(child.state[0])
        sim_copy_temp.game_state.decode_state = sim_copy_temp.game_state.make_state()
        cur_decode_state = sim_copy_temp.game_state.decode_state
        heuval = 0
        if pidx == 0:
            for rel, dec in enumerate(cur_decode_state[:sim_copy_temp.game_state.N_BLOCKS_PER+1]):
                (col, row) = dec
                dist = sim_copy_temp.game_state.N_ROWS - row - 1
                alpha = ALPHA if rel == self.sim.game_state.N_BLOCKS_PER else 1  # more weight to ball
                heuval += alpha * dist
        else:
            for rel, dec in enumerate(cur_decode_state[sim_copy_temp.game_state.N_BLOCKS_PER+1:]):
                (col, row) = dec
                dist = row
                alpha = ALPHA if rel == self.sim.game_state.N_BLOCKS_PER else 1  # more weight to ball
                heuval += alpha * dist
        return 1/heuval

    def _best_child_node_select(self):
        best_score = -1
        best_child = None
        win_length = math.inf
        for child in self.children:
            BETA = self.BETA
            if self.selection_heu:
                sim_temp = deepcopy(self.sim)
                sim_temp.game_state.state = np.array(child.state[0])
                sim_temp.game_state.decode_state = sim_temp.game_state.make_state()
                if sim_temp.game_state.is_termination_state():
                    return child  # best child as terminal state and so its winning
                BETA += self._my_sigmoid(self._selection_heuristic(child, self.state[1], sim_temp, ALPHA=self.ALPHA))
            score = self._UCB1formula(child.wins[self.state[1]], child.num_visits, self.num_visits, BETA)
            if score > best_score or (score == best_score and child.shortest_win_seen[self.state[1]] < win_length):
                best_score = score
                best_child = child
                win_length = child.shortest_win_seen[self.state[1]]
        return best_child

    def _best_child_node_play(self):
        # sometimes do random (just to explore backward moves etc)
        if random.random() < self.EPS:
            print("------------------ Doing random move ------------------")
            return random.choice(self.children)
        best_score = -1  # basically is win percentage here
        best_heu = math.inf
        best_child = None
        win_length = math.inf
        for child in self.children:
            BETA = 2  # only exploit
            sim_temp = deepcopy(self.sim)
            sim_temp.game_state.state = np.array(child.state[0])
            sim_temp.game_state.decode_state = sim_temp.game_state.make_state()
            if sim_temp.game_state.is_termination_state():
                return child  # best child as terminal state and so its winning
            score = self._UCB1formula(child.wins[self.state[1]], child.num_visits, self.num_visits, BETA) / BETA
            heu = 1 / self._selection_heuristic(child, self.state[1], sim_temp, ALPHA=self.ALPHA)
            if score > best_score or (score == best_score and child.shortest_win_seen[self.state[1]] < win_length) or (score == best_score and child.shortest_win_seen[self.state[1]] < 1.5*win_length and heu < best_heu):
                best_score = score
                best_heu = heu
                best_child = child
                win_length = child.shortest_win_seen[self.state[1]]
        return best_child

    def select(self):
        self._update_sim_state(self.state)
        if self.sim.game_state.is_termination_state():
            return self, None  # terminal state
        p = self.state[1]
        if p == self.mcts_idx or self.opponent_policy == "mcts":
            if len(self.untried_actions) > 0:
                act = random.choice(list(self.untried_actions))
                self.untried_actions.discard(act)
                return self, act
            else:
                bc = self._best_child_node_select()
                return bc.select()
        else:
            if self.opponent_policy == "random" or self.opponent_policy == "passive":
                if len(self.untried_actions) > 0:
                    act = random.choice(list(self.untried_actions))
                    self.untried_actions.discard(act)
                    return self, act
                else:
                    random_child = random.choice(self.children)
                    return random_child.select()
            else:
                raise ValueError("Invalid opponent policy")

    def expand(self, action, rules_ball_actions):
        if action is None:
            return self  # terminal state
        self._update_sim_state(self.state)
        self.sim.update(action, self.state[1])
        next_state = tuple((tuple(self.sim.game_state.state), (self.state[1] + 1) % 2))
        if next_state[1] != self.mcts_idx and self.opponent_policy == "passive":
            # only ball actions
            b_copy = deepcopy(self.sim.game_state)
            ball_actions = rules_ball_actions(b_copy, next_state[1])
            ball_actions = [(self.sim.game_state.N_BLOCKS_PER, e) for e in ball_actions]
            next_actions = set(ball_actions)
        else:
            next_actions = self.sim.generate_valid_actions(next_state[1])
        child = MCTSTreeNode(next_state, self.sim, next_actions, parent=self, parent_action=action, cycle=self.cycle, rollout_heu=self.rollout_heu, selection_heu=self.selection_heu, EPS=self.EPS, BETA=self.BETA, ALPHA=self.ALPHA, mcts_idx=self.mcts_idx, opponent_policy=self.opponent_policy)
        self.children.append(child)
        return child

    def rollout(self, playout, rules_ball_actions):
        self._update_sim_state(self.state)
        p = self.state[1]
        path_length = 0
        while not self.sim.game_state.is_termination_state():
            actions = self.sim.generate_valid_actions(p)  # actions is a set
            if p == self.mcts_idx or self.opponent_policy == "mcts":
                if self.cycle:
                    act = self._cycle_detection_get_action(playout, actions, deepcopy(self.sim), p)
                else:
                    act, _ = self._rollout_get_action(actions, deepcopy(self.sim), p)
            else:
                if self.opponent_policy == "random":
                    act = random.choice(list(actions))
                elif self.opponent_policy == "passive":
                    b_copy = deepcopy(self.sim.game_state)
                    ball_actions = rules_ball_actions(b_copy, p)
                    ball_actions = [(self.sim.game_state.N_BLOCKS_PER, e) for e in ball_actions]
                    act = random.choice(ball_actions)
                else:  # same as self (mcts)
                    raise ValueError("Invalid opponent policy")
            self.sim.update(act, p)
            path_length += 1
            p = (p+1) % 2
        winner_idx = (p+1) % 2  # last mover
        self.shortest_win_seen[winner_idx] = min(self.shortest_win_seen[winner_idx], path_length)
        return winner_idx

    def _cycle_detection_get_action(self, playout, actions, sim_copy, pidx):
        avoid_list = []
        while True:
            act, success = self._rollout_get_action(actions, deepcopy(sim_copy), pidx, avoid_list)
            if len(playout.game) == 0 or not success:
                # no moves have been played yet, so no cycle OR no action can be found avoiding the cycle
                return act
            sim_copy_temp = deepcopy(sim_copy)
            sim_copy_temp.update(act, pidx)
            cur_state = tuple((tuple(sim_copy_temp.game_state.state), (pidx + 1) % 2))
            for past_node in reversed(playout.game):
                past_state = past_node.state
                if past_state == cur_state:
                    # Cycle detected
                    avoid_list.append(act)
                    act = None
                    break
            if act is not None:
                return act

    def _distance_to_opposite_end(self, actions, sim_copy, pidx, ALPHA):
        # Calculate the distances to the opposite end of the board of the resulting states due to all actions
        heuvals = {}
        for act in actions:
            heuvals[act] = 0
            sim_copy_temp = deepcopy(sim_copy)
            sim_copy_temp.update(act, pidx)
            cur_decode_state = sim_copy_temp.game_state.decode_state
            if pidx == 0:
                for rel, dec in enumerate(cur_decode_state[:sim_copy_temp.game_state.N_BLOCKS_PER+1]):
                    (col, row) = dec
                    dist = sim_copy_temp.game_state.N_ROWS - row - 1
                    alpha = ALPHA if rel == self.sim.game_state.N_BLOCKS_PER else 1  # more weight to ball
                    heuvals[act] += alpha * dist
            else:
                for rel, dec in enumerate(cur_decode_state[sim_copy_temp.game_state.N_BLOCKS_PER+1:]):
                    (col, row) = dec
                    dist = row
                    alpha = ALPHA if rel == self.sim.game_state.N_BLOCKS_PER else 1  # more weight to ball
                    heuvals[act] += alpha * dist
        return heuvals

    def _rollout_heuristic(self, actions, sim_copy, pidx):
        # Finds rollout heuristic for all actions (heuristic less means better action)
        heuvals = self._distance_to_opposite_end(actions, sim_copy, pidx, ALPHA=self.ALPHA)
        for act in actions:
            sim_copy_temp = deepcopy(sim_copy)
            sim_copy_temp.update(act, pidx)
            if sim_copy_temp.game_state.is_termination_state():
                heuvals[act] -= 10000  # if the action leads to a terminal state, it is the best action
        return heuvals

    def _rollout_get_action(self, actions, sim_copy, pidx, avoid_list=None):
        if avoid_list is None:
            avoid_list = []
        act = random.choice(list(actions))  # to store atleast one action if actions gets empty later
        for act in avoid_list:
            actions.discard(act)
        if len(actions) <= 0:
            # impossible to avoid all the actions in the avoid_list
            return act, False
        if self.rollout_heu and random.random() > self.EPS:  # because sometimes do random
            heuvals = self._rollout_heuristic(actions, sim_copy, pidx)
            act = min(heuvals, key=heuvals.get)
            return act, True
        return random.choice(list(actions)), True

    def backpropagate(self, winner):
        self.num_visits += 1
        self.wins[winner] += 1
        if self.parent is not None:
            self.parent.shortest_win_seen[winner] = min(self.parent.shortest_win_seen[winner], self.shortest_win_seen[winner]+1)
            self.parent.backpropagate(winner)

    def best_action(self):
        bc = self._best_child_node_play()
        return bc.parent_action, bc.wins[self.state[1]]/bc.num_visits
