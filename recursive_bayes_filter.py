import numpy as np
from copy import deepcopy


class ParticleFilter:
    def __init__(self, player_idx, initial_dstate, sim, method="dirac"):
        self.pid = player_idx  # self player's idx (used to compare idx in the state of the particle to see whose move it is)
        self.weighted_particles = {}  # {dstate: weight} where weight is the probability of the state, and state is a gsp_type state
        self.initial_dstate = initial_dstate
        self.sim = sim
        self._initializer(deepcopy(self.sim), method=method)
        if self.pid == 0:
            self.prev_obs = None
        else:
            self.prev_obs = initial_dstate[0]

    def _initializer(self, sim, method="dirac"):
        if method == "dirac":
            self.weighted_particles[self.initial_dstate] = 1.0  # assuming each player precisely knows the initial state
        elif method == "random":
            # generating noisy particles
            N_RANDOM_PARTICLES = 1000
            opponent_idx = (self.pid + 1) % 2
            self.weighted_particles[self.initial_dstate] = 1.0 / (N_RANDOM_PARTICLES + 1)
            for i in range(N_RANDOM_PARTICLES):
                observation = sim.sample_observation(opponent_idx)
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

    def _update_sim_state(self, state):
        self.sim.game_state.state = np.array(state[0])
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

    def _normalize_weights(self, weighted_particles):
        total_weight = sum(weighted_particles.values())
        if total_weight == 1.0:  # FOR_SPEED
            return weighted_particles
        return {state: weight / total_weight for state, weight in weighted_particles.items()}

    def get_belief_MLE(self):
        b = self.weighted_particles
        m = max(b, key=b.get)
        return m, b[m], len(self.weighted_particles)

    def belief_update_observation(self, new_obs):
        next_weighted_particles = {}
        # check if prev obs is None, then it means initial state so exclude the action term in weight calculation
        if self.prev_obs is not None:
            possible_pieces = self.get_possible_pieces_opponent_moved(new_obs, self.prev_obs)

        for dstate, weight in self.weighted_particles.items():
            assert weight > 0, "MYERROR: weight is not positive"
            encoded_state = tuple(self.sim.game_state.encode_single_pos(s) for s in dstate[0])
            state = tuple((encoded_state, dstate[1]))
            self._update_sim_state(state)
            if self.prev_obs is not None:
                assert (self.pid + 1) % 2 == dstate[1], "MYERROR: player_idx is not correct"
                U = self.sim.generate_valid_actions(dstate[1], only_pieces=possible_pieces)
                for u in U:
                    self._update_sim_state(state)
                    self.sim.update(u, dstate[1])
                    next_dstate = tuple((tuple(self.sim.game_state.decode_state), (dstate[1]+1) % 2))
                    p_obs = self.observation_model_state(new_obs, next_dstate, dstate[1])
                    w = p_obs * (1/len(U)) * weight
                    if w > 0:
                        if next_dstate in next_weighted_particles.keys():
                            next_weighted_particles[next_dstate] += w
                        else:
                            next_weighted_particles[next_dstate] = w
            else:
                self._update_sim_state(state)
                p_obs = self.observation_model_state(new_obs, dstate, (dstate[1] + 1) % 2)
                w = p_obs * weight
                if w > 0:
                    next_weighted_particles[dstate] = w
        self.weighted_particles = self._normalize_weights(next_weighted_particles)

    def observation_model_state(self, obs, dstate, opponent_idx):
        tot_prob = 1.0
        # loop over opponent pieces only
        for rel_i in range(self.sim.game_state.N_BLOCKS_PER+1):
            abs_i = self.get_abs_from_rel_index(rel_i, opponent_idx)
            p_obs = self.observation_model_state_piece(obs, dstate, abs_i)
            tot_prob *= p_obs

        # DEBUG: remove later
        for rel_i in range(self.sim.game_state.N_BLOCKS_PER+1):
            abs_i = self.get_abs_from_rel_index(rel_i, (opponent_idx+1) % 2)
            assert obs[abs_i] == dstate[0][abs_i], "MYERROR: observation is not correct"

        return tot_prob

    def observation_model_state_piece(self, obs, dstate, abs_i):
        obs_pos = obs[abs_i]
        actual_pos = dstate[0][abs_i]
        obs_model, possible_obs_positions = self.sim.modify_observation_model(abs_i, dstate[0])
        if obs_pos not in possible_obs_positions:
            return 0.0
        return float(obs_model[possible_obs_positions.index(obs_pos)])

    def process_feedback_invalid_action(self, u, new_obs):
        next_weighted_particles = {}
        for dstate, weight in self.weighted_particles.items():
            assert weight > 0, "MYERROR: weight is not positive"
            encoded_state = tuple(self.sim.game_state.encode_single_pos(s) for s in dstate[0])
            assert dstate[1] == self.pid, "MYERROR: player_idx is not correct"
            state = tuple((encoded_state, dstate[1]))
            self._update_sim_state(state)
            try:
                is_valid_action = self.sim.validate_action(u, dstate[1])
            except ValueError:
                is_valid_action = False
            if not is_valid_action:
                next_weighted_particles[dstate] = weight
        self.weighted_particles = self._normalize_weights(next_weighted_particles)

    def belief_propagate_own_action(self, u, new_obs):
        next_weighted_particles = {}
        for dstate, weight in self.weighted_particles.items():
            assert weight > 0, "MYERROR: weight is not positive"
            encoded_state = tuple(self.sim.game_state.encode_single_pos(s) for s in dstate[0])
            assert dstate[1] == self.pid, "MYERROR: player_idx is not correct"
            state = tuple((encoded_state, dstate[1]))
            self._update_sim_state(state)
            try:
                is_valid_action = self.sim.validate_action(u, dstate[1])
            except ValueError:
                is_valid_action = False
            if is_valid_action:
                self._update_sim_state(state)
                self.sim.update(u, dstate[1])
                next_dstate = tuple((tuple(self.sim.game_state.decode_state), (dstate[1]+1) % 2))
                next_weighted_particles[next_dstate] = weight
        self.weighted_particles = self._normalize_weights(next_weighted_particles)
        self.prev_obs = new_obs

    def my_heuristic_get_possible_pieces_opponent_moved(self, new_obs, prev_obs):
        # BUG: when observation such that multiple pieces at ball's same location -> SOL don't remove that idx in that case
        # BUG: ball moved condition works only if above BUG is not there
        def check1(new_obs, prev_obs, abs_i):
            return new_obs[abs_i] == prev_obs[abs_i]

        def check2(new_obs, prev_obs, abs_i):
            new_pos = np.array(new_obs[abs_i])
            prev_pos = np.array(prev_obs[abs_i])
            return np.sum(np.abs(new_pos-prev_pos)) == 1

        def check3or4(new_obs, prev_obs, abs_i):
            new_pos = np.array(new_obs[abs_i])
            prev_pos = np.array(prev_obs[abs_i])
            return np.sum(np.abs(new_pos-prev_pos)) == 2

        opponent_idx = (self.pid + 1) % 2
        possible_pieces = set()
        for rel_i in range(self.sim.game_state.N_BLOCKS_PER+1):
            possible_pieces.add(rel_i)
        opp_ball_idx = self.get_abs_from_rel_index(self.sim.game_state.N_BLOCKS_PER, opponent_idx)
        prev_ball_on_block_idx = None
        new_ball_on_block_idx = None

        for rel_i in range(self.sim.game_state.N_BLOCKS_PER):
            abs_i = self.get_abs_from_rel_index(rel_i, opponent_idx)
            if prev_obs[abs_i] == prev_obs[opp_ball_idx]:
                prev_ball_on_block_idx = rel_i
            if new_obs[abs_i] == new_obs[opp_ball_idx]:
                new_ball_on_block_idx = rel_i

        assert prev_ball_on_block_idx is not None and new_ball_on_block_idx is not None, "MYERROR: Ball not on block in either prev or new obs"
        possible_pieces.remove(prev_ball_on_block_idx)  # the block that the ball was on before is not possible to be moved
        if prev_ball_on_block_idx != new_ball_on_block_idx:
            return set([self.sim.game_state.N_BLOCKS_PER])  # ball moved
        possible_pieces.remove(self.sim.game_state.N_BLOCKS_PER)  # ball did not move
        possible_pieces_copy = deepcopy(possible_pieces)
        for rel_i in possible_pieces_copy:
            abs_i = self.get_abs_from_rel_index(rel_i, opponent_idx)
            if check1(new_obs, prev_obs, abs_i):
                possible_pieces.remove(rel_i)
            elif check2(new_obs, prev_obs, abs_i) or check3or4(new_obs, prev_obs, abs_i):
                continue
            else:
                return set([rel_i])  # this piece moved!
        return possible_pieces

    def get_possible_pieces_opponent_moved(self, new_obs, prev_obs):
        opponent_idx = (self.pid + 1) % 2
        possible_pieces = set()
        for rel_i in range(self.sim.game_state.N_BLOCKS_PER+1):
            possible_pieces.add(rel_i)
        return possible_pieces

    def get_abs_from_rel_index(self, rel_i, player_i):
        return (self.sim.game_state.N_BLOCKS_PER+1)*player_i + rel_i

    def get_rel_from_abs_index(self, abs_i, player_i):
        return abs_i - (self.sim.game_state.N_BLOCKS_PER+1)*player_i
