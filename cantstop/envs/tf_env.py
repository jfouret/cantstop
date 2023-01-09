from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from cantstop.envs.constants import \
    ACTION_ENCODING, FIRST_PAIR_ENCODING, SECOND_PAIR_ENCODING, B_HIGH
from tf_agents.typing import types


class TfPyEnv(py_environment.PyEnvironment):

    def __init__(self, n_players: int):
        self._num_players = n_players
        obs_dim = 4 + (n_players + 1) * 11
        max_obs = np.array(
          4 * [5] + [13 - 2 * abs(i - 5) for i in range(11)] * (n_players + 1),
          dtype=np.uint8)
        # ACTION is of size 12
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(1,), dtype=np.uint8, minimum=0, maximum=11, name="action")
        self._observation_spec = {
          "observation": array_spec.BoundedArraySpec(
            shape=(obs_dim,), dtype=np.uint8,
            minimum=np.zeros((obs_dim,), dtype=np.uint8),
            maximum=max_obs, name="observation"),
          "mask": array_spec.ArraySpec(
              shape=(12,), dtype=np.bool_, name="mask")}
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # The board represents the paths advancment for all players.
        # The first dimension represent players.
        # The second dimension represent the paths.
        # Values are between zero and one.
        self._board = np.zeros((self._num_players, 11), dtype=np.uint8)
        self._turn = 0
        # current phase
        # 0: The player should decide whether to play or pass
        # 1: dices have been thrown
        self._value_mask = np.array([True]*11, dtype=np.bool_)
        self._unsaved = np.zeros(11, dtype=np.uint8)
        self._throw_dices()
        return ts.restart({
                    "observation": np.concatenate((
                        self._unsaved,
                        self._board.flatten(),
                        self._dice_values)),
                    "mask": np.array([True]*11, dtype=np.bool_)
                })

    def seed(self, seed: types.Seed) -> None:
        self.np_random = np.random.default_rng(seed=seed)

    def _throw_dices(self) -> None:
        self._dice_values = \
            self.np_random.integers(np.ones(4)*5, dtype=np.uint8)

    def _step(self, action: types.Array):

        dice_comb, play_or_pass = ACTION_ENCODING[action[0]]

        value_mask = np.logical_and(
          self._value_mask,
          self._board[0, ] + self._unsaved != B_HIGH
        )
        # get first dice comb value
        first_value = np.sum(
            self._dice_values[FIRST_PAIR_ENCODING[dice_comb]])
        # increment score or raise error
        if value_mask[first_value]:
            raise ValueError("Illegal action")
        self._unsaved[first_value] += 1
        # update mask
        value_mask = np.logical_and(
          value_mask,
          self._unsaved + self._board[0, ] != B_HIGH
        )

        # get second dice comb value
        second_value = np.sum(
            self._dice_values[SECOND_PAIR_ENCODING[dice_comb]])
        if value_mask[second_value]:
            self._unsaved[second_value] += 1
            # update mask
            value_mask = np.logical_and(
                value_mask,
                self._unsaved + self._board[0, ] != B_HIGH
            )

        if np.sum(self._unsaved + self._board[0, ] != B_HIGH) > 2:
            reward = 1
            return ts.termination(
                {
                    "observation": np.concatenate((
                        self._unsaved,
                        self._board.flatten(),
                        self._dice_values)),
                    "mask": np.array([False]*11, dtype=np.bool_)
                },
                reward)

        if play_or_pass == 0:
            # save and stop
            self._value_mask = value_mask
            self._board[0, ] = self._unsaved + self._board[0, ]
            self._unsaved = np.zeros(11, dtype=np.uint8)
            self._board = np.roll(self._board, 1, axis=0)
            self._turn += 1
            reward = -1
        elif play_or_pass == 1:
            reward = 0
        else:
            raise ValueError("Unrecognized action")

        any_legal_value = False
        while not any_legal_value:
            self._throw_dices()
            comb_mask = \
                [value_mask[np.sum(self._dice_values[FIRST_PAIR_ENCODING[i]])]
                 for i in range(6)]
            any_legal_value = np.any(comb_mask)
            if not any_legal_value:
                reward = -1
                self._unsaved = np.zeros(11, dtype=np.uint8)
                self._board = np.roll(self._board, 1, axis=0)
                self._turn += 1

        obs = {
            "observation": np.concatenate((
                self._unsaved,
                self._board.flatten(),
                self._dice_values)),
            "mask": comb_mask * 2
            }

        return ts.transition(obs, reward)
