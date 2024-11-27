import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from .TSP.pytorch.NNetWrapper import NNetWrapper

from TSP.TSPGame import TSPGame

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: TSPGame, nnet: NNetWrapper, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play.
        As the game is played, each move is added as a training example to
        trainExamples. The game is played until a maximum number of steps is reached.
        After the episode ends, the final tour length is used to assign values to each example.
        Returns:
            trainExamples: a list of examples of the form (board, pi, v)
                        pi is the MCTS-informed policy vector, v is the negative tour length.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        episodeStep = 0

        maxSteps = self.args.maxSteps  # Define this in your arguments

        while episodeStep < maxSteps:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            # Apply data augmentation by rearranging node and tour order
            symmetries = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in symmetries:
                trainExamples.append([b, p, None])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.getNextState(board, action)

        # After the episode ends, compute the final tour length
        final_tour_length = self.game.getTourLength(board)
        value = -final_tour_length  # Negative tour length as value to minimize

        # Assign the value to all examples
        return [(x[0], x[1], value) for x in trainExamples]


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains the neural network with
        examples in trainExamples (which has a maximum length of maxlenOfQueue).
        Since TSP is a single-player game, we can evaluate the new network
        based on the improvement in tour length.
        """
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")

            iterationTrainExamples = []

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # Reset search tree
                iterationTrainExamples.extend(self.executeEpisode())

            # Save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(f"Removing the oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # Training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("EVALUATING NEW NETWORK")
            # Evaluate the new network by comparing average tour lengths
            avg_length_old = self.evaluateNetwork(pmcts)
            avg_length_new = self.evaluateNetwork(nmcts)

            log.info(f"Average Tour Length - New: {avg_length_new}, Old: {avg_length_old}")

            if avg_length_new < avg_length_old * (1 - self.args.updateThreshold):
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    def evaluateNetwork(self, mcts):
        """
        Evaluates the network by running it over a number of episodes and computing
        the average tour length.
        """
        total_length = 0
        num_episodes = self.args.numEpsEval  # Define this in your arguments
        for _ in range(num_episodes):
            board = self.game.getInitBoard()
            for _ in range(self.args.maxSteps):
                canonicalBoard = self.game.getCanonicalForm(board)
                pi = mcts.getActionProb(canonicalBoard, temp=0)
                action = np.argmax(pi)
                board = self.game.getNextState(board, action)
            total_length += self.game.getTourLength(board)
        avg_length = total_length / num_episodes
        return avg_length


    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_examples_folder_file[0], self.args.load_examples_folder_file[1]
        )
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Loading done!")

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
