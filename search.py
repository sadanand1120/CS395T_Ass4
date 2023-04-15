import numpy as np
from game import BoardState, GameSimulator, Rules, VanillaMCTSPlayer, RandomPlayer, PassivePlayer
from game import ProbabilisticRandomPlayer, ProbabilisticVanillaMCTSPlayer
from MCTS import MCTSTreeNode, MCTSPlayout, MCTSPlayRecordedGame
from board_gui import EOTgui
from matplotlib import pyplot as plt

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set


class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        The form of initial state is:
        ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
        """
        if not (initial_board_state.is_valid() and goal_board_state.is_valid()):
            raise ValueError("Invalid Initial or Goal Board configuration!")
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * (self.sim.game_state.N_BLOCKS_PER+1)
        return tuple((tuple(s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    def vanilla_mcts(self, state, playout, iters, cycle, rollout_heu, selection_heu, opponent_policy, EPS, ALPHA, BETA):
        """
        A vanilla Monte Carlo Tree Search implementation with:
            (1) Selection of an action based on UCB1 (exploitation + exploration)
            (2) Expansion of a node (basically when the result of action selection through UCB1 leads to opening up an unexplored node)
            (3) Simulation of a random rollout until terminal state is reached
            (4) Backpropagation of the result of the rollout from expanded node upto the root of the tree
        """
        # if playout's last node has a child that matches the state, then do NOT insert a new node
        root = MCTSTreeNode(state, self.sim, self.get_actions(state), cycle=cycle, rollout_heu=rollout_heu, selection_heu=selection_heu, EPS=EPS, ALPHA=ALPHA, BETA=BETA, mcts_idx=state[1], opponent_policy=opponent_policy)
        if len(playout.game) > 0:
            for child in playout.game[-1].children:
                if child.state == state:
                    root = child
                    # root.parent = None  # not required actually, let it go through upto the original root (because thats required in MCTSPlayRecordedGame)
                    break
        playout.add_node(root)

        for i in range(iters):
            print(f"Doing iteration: {i}")
            if i < iters-1:
                print(LINE_UP, end=LINE_CLEAR)
            snode, action = root.select()
            enode = snode.expand(action, rules_ball_actions=Rules.single_ball_actions)
            winner = enode.rollout(playout, rules_ball_actions=Rules.single_ball_actions)
            enode.backpropagate(winner)
        return root.best_action()  # action, value of the resulting state


if __name__ == "__main__":
    for i in range(71, 72, 1):
        e = EOTgui(tsec=2)
        e.initialize_pieces()
        b1 = BoardState()
        playout = MCTSPlayout()
        ini_dstate = tuple((tuple(b1.decode_state), 0))
        # the player_idx can be set to 0 in gsp for both players because its not used anywhere - might as well set 0 and 1
        players = [
            ProbabilisticRandomPlayer(GameStateProblem(b1, b1, 0), 0, playout, ini_dstate, method="random"),
            # ProbabilisticVanillaMCTSPlayer(GameStateProblem(b1, b1, 0), 0, playout, ini_dstate, method="random", iters=1000, opponent_policy="mcts", EPS=0.04),
            ProbabilisticRandomPlayer(GameStateProblem(b1, b1, 0), 1, playout, ini_dstate, method="random")
            # ProbabilisticVanillaMCTSPlayer(GameStateProblem(b1, b1, 0), 1, playout, ini_dstate, method="random", iters=1000, opponent_policy="mcts", EPS=0.04)
        ]
        sim = GameSimulator(players, gui=e)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        print("Winner: ", winner)
        print("Rounds: ", rounds)
        plt.close("all")

        # input("Press Enter to start playing recorded game (what players had their MLE states as)...")

        # rec = MCTSPlayRecordedGame(playout, list(playout.game[0].state[0]), tsec=4)
        # rec.play()
