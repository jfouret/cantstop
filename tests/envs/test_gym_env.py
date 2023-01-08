import pytest

import numpy as np
from cantstop.envs import GymEnv
from cantstop.agents import RuleBasedAgent
from cantstop.envs.constants import B_HIGH


@pytest.mark.parametrize("num_players,board_one_hot,dices_one_hot", [
    (1, False, False),
    (1, False, True),
    (1, True, False),
    (1, True, True),
    (2, False, False),
    (2, True, False),
    (2, False, True),
    (2, True, True),
    (3, False, False),
    (3, True, False),
    (3, False, True),
    (3, True, True)
])
def test_dimensions(num_players, board_one_hot, dices_one_hot):
    env = GymEnv(
        num_players=num_players,
        board_one_hot=board_one_hot,
        dices_one_hot=dices_one_hot)
    state = env.reset()
    if dices_one_hot:
        dices_dims = 4 * 6
    else:
        dices_dims = 4
    if board_one_hot:
        board_dims = (num_players + 1) * np.sum(B_HIGH)
    else:
        board_dims = (num_players + 1) * 11
    dims = board_dims + dices_dims
    assert state.shape == (dims,)


@pytest.mark.parametrize("num_players,board_one_hot,dices_one_hot", [
    (1, False, False),
    (1, True, True),
    (3, True, False),
    (3, True, True)
])
def test_with_rule_based_agent(num_players, board_one_hot, dices_one_hot):
    env = GymEnv(
        num_players=num_players,
        board_one_hot=board_one_hot,
        dices_one_hot=dices_one_hot
    )
    agent = RuleBasedAgent()
    t1 = agent.play_a_lot(env, seed=0, n_play=10)
    t2 = agent.play_a_lot(env, seed=1, n_play=10)
    t3 = agent.play_a_lot(env, seed=0, n_play=10)
    assert np.all(t1["turns"] == t3["turns"])
    assert np.any(t2["turns"] != t1["turns"])
    assert np.all(t1["total_rewards"] == t3["total_rewards"])
    assert np.any(t2["total_rewards"] != t1["total_rewards"])
    assert np.mean(t1["turns"]) < 20
    assert np.max(t1["turns"]) < 25
