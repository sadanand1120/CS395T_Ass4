import numpy as np
from copy import deepcopy
from game import GameSimulator, BoardState

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


class Q1:
    def __init__(self, NUM, initial_dstate, sim, method="dirac"):
        self.ground_truth_seq = []  # list of dstates
        self.ground_truth_act_seq = []  # list of actions
        self.obs_seq = []  # list of dstates
        self.weighted_particles = {}  # {dstate: weight} where weight is the probability of the state, and state is a gsp_type state
        self.initial_dstate = initial_dstate
        self.sim = sim
        self._initializer(deepcopy(self.sim), method=method)
        self.weighted_actions = {}  # {action: weight} where weight is the probability that this action was taken, and action has format (player_idx, rel_idx, enc, dec)
        self.prev_obs = None
        self.NUM = NUM
        self.read_seq()

    def _initializer(self, sim, method="dirac"):
        if method == "dirac":
            self.weighted_particles[self.initial_dstate] = 1.0  # assuming each player precisely knows the initial state
        elif method == "random":
            # generating noisy particles
            N_RANDOM_PARTICLES = 1000
            self.weighted_particles[self.initial_dstate] = 1.0 / (N_RANDOM_PARTICLES + 1)
            for i in range(N_RANDOM_PARTICLES):
                observation0 = sim.sample_observation(0)
                observation1 = sim.sample_observation(1)
                observation = deepcopy(observation0)
                observation[sim.game_state.N_BLOCKS_PER+1:] = observation1[sim.game_state.N_BLOCKS_PER+1:]
                obs_dstate = tuple((tuple(observation), self.initial_dstate[1]))
                if obs_dstate in self.weighted_particles.keys():
                    self.weighted_particles[obs_dstate] += 1.0 / (N_RANDOM_PARTICLES + 1)
                else:
                    self.weighted_particles[obs_dstate] = 1.0 / (N_RANDOM_PARTICLES + 1)
                encoded_state = tuple(sim.game_state.encode_single_pos(s) for s in observation)
                if i % 100 == 0:
                    sim.game_state.state = np.array(encoded_state)
                    sim.game_state.decode_state = sim.game_state.make_state()
        else:
            raise ValueError("MYERROR: invalid method of initialization")

    def read_seq(self):
        gtpath = f"data/ground_truth_sequences/{self.NUM}.txt"
        obspath = f"data/observed_sequences/{self.NUM}.txt"
        actionpath = f"data/ground_truth_action_sequences/{self.NUM}.txt"
        with open(gtpath, "r") as f:
            self.ground_truth_seq = [eval(line) for line in f.readlines()]
        with open(obspath, "r") as f:
            self.obs_seq = [eval(line) for line in f.readlines()]
        with open(actionpath, "r") as f:
            self.ground_truth_act_seq = [eval(line) for line in f.readlines()]

    def write_belief(self):
        bpath = f"data/belief_sequences/{self.NUM}.txt"
        with open(bpath, "a") as f:
            bel = self.get_belief_MLE()
            f.write(f"{self.weighted_particles} ==> {[bel[0], bel[1]]}\n")

    def write_action(self):
        actionpath = f"data/inferred_action_sequences/{self.NUM}.txt"
        with open(actionpath, "a") as f:
            f.write(f"{self.weighted_actions}\n")

    def write_ground_truth_action(self, cur, nxt):
        actionpath = f"data/ground_truth_action_sequences/{self.NUM}.txt"
        pid = cur[1]
        dec = None
        for i in range(len(cur[0])):
            if cur[0][i] != nxt[0][i]:
                rel_idx = self.get_rel_from_abs_index(i, i // (self.sim.game_state.N_BLOCKS_PER+1))
                dec = nxt[0][i]
                break
        action = (pid, rel_idx, self.sim.game_state.encode_single_pos(dec), dec)
        with open(actionpath, "a") as f:
            f.write(f"{action}\n")

    def get_abs_from_rel_index(self, rel_i, player_i):
        return (self.sim.game_state.N_BLOCKS_PER+1)*player_i + rel_i

    def get_rel_from_abs_index(self, abs_i, player_i):
        return abs_i - (self.sim.game_state.N_BLOCKS_PER+1)*player_i

    def _normalize_weights(self, weighted_particles):
        total_weight = sum(weighted_particles.values())
        if total_weight == 1.0:  # FOR_SPEED
            return weighted_particles
        return {state: weight / total_weight for state, weight in weighted_particles.items()}

    def get_possible_pieces_opponent_moved(self):
        possible_pieces = set()
        for rel_i in range(self.sim.game_state.N_BLOCKS_PER+1):
            possible_pieces.add(rel_i)
        return possible_pieces

    def _update_sim_state(self, state):
        self.sim.game_state.state = np.array(state[0])
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

    def observation_model_state(self, obs, dstate):
        tot_prob = 1.0
        for abs_i in range(len(self.sim.game_state.state)):
            p_obs = self.observation_model_state_piece(obs, dstate, abs_i)
            tot_prob *= p_obs
        return tot_prob

    def observation_model_state_piece(self, obs, dstate, abs_i):
        obs_pos = obs[abs_i]
        actual_pos = dstate[0][abs_i]
        obs_model, possible_obs_positions = self.sim.modify_observation_model(abs_i, dstate[0])
        if obs_pos not in possible_obs_positions:
            return 0.0
        return float(obs_model[possible_obs_positions.index(obs_pos)])

    def get_belief_MLE(self):
        b = self.weighted_particles
        m = max(b, key=b.get)
        return m, b[m]
    
    def get_action_MLE(self):
        b = self.weighted_actions
        m = max(b, key=b.get)
        return m, b[m]

    def belief_update_observation(self, new_obs, new_obs_idx):
        next_weighted_actions = {}
        next_weighted_particles = {}
        if self.prev_obs is not None:
            possible_pieces = self.get_possible_pieces_opponent_moved()

        for dstate, weight in self.weighted_particles.items():
            assert weight > 0, "MYERROR: weight is not positive"
            encoded_state = tuple(self.sim.game_state.encode_single_pos(s) for s in dstate[0])
            state = tuple((encoded_state, dstate[1]))
            self._update_sim_state(state)
            if self.prev_obs is not None:
                assert (new_obs_idx + 1) % 2 == dstate[1], "MYERROR: new_obs_idx is not correct"
                U = self.sim.generate_valid_actions(dstate[1], only_pieces=possible_pieces)
                for u in U:
                    self._update_sim_state(state)
                    self.sim.update(u, dstate[1])
                    next_dstate = tuple((tuple(self.sim.game_state.decode_state), (dstate[1]+1) % 2))
                    p_obs = self.observation_model_state(new_obs, next_dstate)
                    w = p_obs * (1/len(U)) * weight
                    if w > 0:
                        if next_dstate in next_weighted_particles.keys():
                            next_weighted_particles[next_dstate] += w
                        else:
                            next_weighted_particles[next_dstate] = w
                        action = (dstate[1], u[0], u[1], self.sim.game_state.decode_single_pos(u[1]))
                        if action in next_weighted_actions.keys():
                            next_weighted_actions[action] += w
                        else:
                            next_weighted_actions[action] = w
            else:
                self._update_sim_state(state)
                p_obs = self.observation_model_state(new_obs, dstate)
                w = p_obs * weight
                if w > 0:
                    next_weighted_particles[dstate] = w
        self.weighted_particles = self._normalize_weights(next_weighted_particles)
        self.weighted_actions = self._normalize_weights(next_weighted_actions)
        self.prev_obs = new_obs

    def run_sequence(self):
        for i, obs_dstate in enumerate(self.obs_seq):
            print(f"---- Doing observation {i+1} of {len(self.obs_seq)}")
            if i < len(self.obs_seq) - 1:
                print(LINE_UP, end=LINE_CLEAR)
            self.belief_update_observation(*obs_dstate)
            self.write_belief()
            if i < len(self.obs_seq) - 1:
                self.write_ground_truth_action(self.ground_truth_seq[i], self.ground_truth_seq[i+1])
            if i > 0:
                self.write_action()


if __name__ == "__main__":
    for i in range(1, 71, 1):
        print(f"#################################### Doing file {i}")
        b1 = BoardState()
        ini_dstate = tuple((tuple(b1.decode_state), 0))
        q1 = Q1(i, ini_dstate, GameSimulator(None), method="random")
        q1.run_sequence()
