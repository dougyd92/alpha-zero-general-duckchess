import Arena
from MCTS import MCTS

from duckchess.DuckChessGame import DuckChessGame
from duckchess.DuckChessNetWrapper import NNetWrapper as nn
from duckchess.DuckChessPlayers import *

import logging
import coloredlogs

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

human_vs_cpu = True

g = DuckChessGame()

rp = RandomPlayer(g).play
hp = HumanDuckChessPlayer(g).play

# nnet players
n1 = nn(g)
n1.load_checkpoint(folder='./temp/duckchessv0/', filename='checkpoint_1.pth.tar')

args1 = dotdict({'numMCTSSims': 60, 'cpuct':1.0, 'verbose': True})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = nn(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

# The interface is much easier if the human is white
arena = Arena.Arena(player2, n1p, g, display=(lambda x: x))

print(arena.playGame(verbose=True))
