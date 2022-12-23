import logging

import coloredlogs

from Coach import Coach
from duckchess.DuckChessGame import DuckChessGame
from duckchess.DuckChessNetWrapper import NNetWrapper as nn
from utils import *

# Debug, trying to reproduce a specific error
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 0,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/duckchessv0/',
    'load_model': True,
    'load_folder_file': ('./temp/duckchessv0/','checkpoint_1.pth.tar'),
    'starting_iteration': 2,    # Set to higher than 1 if resuming from a checkpoint
    'numItersForTrainExamplesHistory': 2,
    'verbose': False
})


def main():
    log.info('Loading %s...', DuckChessGame.__name__)
    game = DuckChessGame()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
