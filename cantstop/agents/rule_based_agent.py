from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten


class Rule(Enum):
    # target the best 3 and pass or play randomly and replay
    TARGET_BESTS_AND_PASS = 1


class RuleBasedAgent():
    def __init__(self, rule: Rule = Rule.TARGET_BESTS_AND_PASS):
        self._rule = rule
        self.rng = None

    def reset(self, seed: np.uint32) -> None:
        self._rng = np.random.RandomState(seed)

    def act(self, env: gym.Env) -> np.ndarray[np.float32]:
        if self._rng is None:
            raise AttributeError(
                "Random generator must be initiated with reset method")
        if self._rule == Rule.TARGET_BESTS_AND_PASS:
            action = None
            info = env._get_info()
            for high_value in info["high_values"]:
                if high_value in info["possible_values"].keys():
                    action = flatten(env.action_space, {
                      "pair_of_dice_chosen":
                      self._rng.choice(info["possible_values"][high_value]),
                      "throw_dices": 0
                    })
                    break
            if action is None:
                action = flatten(env.action_space, {
                    "pair_of_dice_chosen":
                    self._rng.choice(
                      info["possible_values"][
                        self._rng.choice(
                            [k for k in info["possible_values"].keys()])]),
                    "throw_dices": 1
                })
            return action
        else:
            AttributeError("Agent's rule not recognized")

    def play_a_lot(self, env: gym.Env, seed: np.uint32, n_play: int) -> dict:
        rng = np.random.RandomState(seed)
        uint_max = np.iinfo(np.uint32).max
        np_tot_reward = np.zeros(n_play)
        np_turn = np.zeros(n_play)
        for i in range(n_play):
            env.reset(seed=int(rng.randint(0, uint_max, dtype=np.uint32)))
            self.reset(seed=int(rng.randint(0, uint_max, dtype=np.uint32)))
            is_terminated = False
            while not is_terminated:
                state, reward, is_terminated, info = env.step(self.act(env))
                np_tot_reward[i] += reward
            np_turn[i] = info["turn"]
        return {
            "total_rewards": np_tot_reward,
            "turns": np_turn
        }
