import Arena
from MCTS import MCTS

from duckchess.DuckChessGame import DuckChessGame
from duckchess.DuckChessNetWrapper import NNetWrapper as nn
from duckchess.DuckChessPlayers import *

import argparse

import logging
import coloredlogs

import numpy as np
from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

"""
Compare model to a baseline agent that makes random moves
"""
def main():
    parser = argparse.ArgumentParser(
        prog='compare_to_random.py',
        description='Evaluate a model against a baseline (random moves)'
    )
    parser.add_argument('model_dir', help="Directory with the saved model")
    parser.add_argument('model_name', help="Name of the file, e.g. 'checkpoint_5.pth.tar'")
    args = parser.parse_args()

    g = DuckChessGame()

    n1 = nn(g)
    n1.load_checkpoint(folder=args.model_dir, filename=args.model_name)

    args1 = dotdict({'numMCTSSims': 20, 'cpuct':1.0, 'verbose': False})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    player2 = RandomPlayer(g).play

    arena = Arena.Arena(n1p, player2, g, display=(lambda x: x))

    oneWon, twoWon, draws = arena.playGames(10, verbose=False)

    print(f"{args.model_name} wins:{oneWon},  losses:{twoWon},  draws:{draws}  (playing against random)")

if __name__ == "__main__":
    main()
