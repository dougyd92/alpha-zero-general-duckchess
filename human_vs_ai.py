import Arena
from MCTS import MCTS

from duckchess.DuckChessGame import DuckChessGame
from duckchess.DuckChessNetWrapper import NNetWrapper as nn
from duckchess.DuckChessPlayers import *

import logging
import coloredlogs

import numpy as np
from utils import *

import argparse

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

def main():
    parser = argparse.ArgumentParser(
        prog='human_vs_ai.py',
        description='Play against a trained model on the commandline'
    )
    parser.add_argument('model_dir', help="Directory with the saved model")
    parser.add_argument('model_name', help="Name of the file, e.g. 'checkpoint_5.pth.tar'")
    args = parser.parse_args()

    g = DuckChessGame()

    player2 = hp = HumanDuckChessPlayer(g).play

    n1 = nn(g)
    n1.load_checkpoint(folder=args.model_dir, filename=args.model_name)

    args1 = dotdict({'numMCTSSims': 60, 'cpuct':1.0, 'verbose': True})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


    # The interface is much easier if the human is white
    arena = Arena.Arena(player2, n1p, g, display=(lambda x: x))
    print("You are playing as white")
    print(arena.playGame(verbose=True))

if __name__ == "__main__":
    main()
