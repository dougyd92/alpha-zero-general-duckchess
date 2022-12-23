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
Compare two models against each other
"""
def main():
    parser = argparse.ArgumentParser(
        prog='head_to_head.py',
        description='Evaluate one model against another'
    )
    parser.add_argument('model1_dir', help="Directory with the saved model, for model 1")
    parser.add_argument('model1_name', help="Name of the file for model 1, e.g. 'checkpoint_0.pth.tar'")
    parser.add_argument('model2dir', help="Directory with the saved model, for model 2")
    parser.add_argument('model2name', help="Name of the file for model 2, e.g. 'checkpoint_5.pth.tar'")    
    args = parser.parse_args()

    g = DuckChessGame()

    n1 = nn(g)
    n2 = nn(g)
    n1.load_checkpoint(folder=args.model1_dir, filename=args.model1_name)
    n2.load_checkpoint(folder=args.model2_dir, filename=args.model2_name)

    args1 = dotdict({'numMCTSSims': 20, 'cpuct':1.0, 'verbose': False})
    args2 = args1
    mcts1 = MCTS(g, n1, args1)
    mcts2 = MCTS(g, n1, args2)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, n2p, g, display=(lambda x: x))

    oneWon, twoWon, draws = arena.playGames(10, verbose=False)

    print(f"{args.model1_name} wins:{oneWon}")
    print(f"{args.model2_name} wins:{twoWon}")
    print(f"draws:{draws}")

if __name__ == "__main__":
    main()
