import logging

import coloredlogs

from Coach import Coach
from mancala.MancalaGame import MancalaGame as Game
from mancala.pytorch.NNet import NNetWrapper as nn
from utils import *
import torch

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 1,  # Number of iterations
        "numEps": 10,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 8,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./mtemp/",
        "load_model": False,  # False originally
        "load_folder_file": ("./mtemp", "mancala_best.pth.tar"),  # dev/models/8x100x50
        "load_examples": False,  # my new addition not yet hooked up yar
        "load_examples_folder_file": ("./mtemp", "mancala_checkpoint_4.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)


def main():
    print(f"Is cuda there yar: {torch.cuda.is_available()}")
    log.info("Loading %s...", Game.__name__)
    g = Game(6)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()