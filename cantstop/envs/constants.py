from numpy import uint8, array, zeros
from enum import Enum

# possible indices for the first pair of dices
FIRST_PAIR_ENCODING = array([
  [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
])

# possible indices for the second pair of dices
# given the first combination
SECOND_PAIR_ENCODING = array([
  [2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]
])

# lower values on the board
B_LOW = zeros(11, dtype=uint8)

# higher values on the board
B_HIGH = array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3], dtype=uint8)


class RewardType(Enum):
    MINUS_ONE_PER_TURN = 1
