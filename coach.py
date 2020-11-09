import numpy as np
import random
import time, os, sys

from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from mcts import MCTS
from pretrainer import normalize_list
from lib.puzzles_generator_MET import prepare_data_p, prepare_data_v, prepare_fragments
from lib.utils import sigmoid, softmax

DATASET_PATH = '../datasets/MET/'

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet_p,nnet_v, args):
        self.game = game
        self.nnet_p = nnet_p
        self.nnet_v = nnet_v
        self.args = args
        self.mcts = MCTS(self.game, self.nnet_p, self.nnet_v, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self, fragments, solution_dict, nb_help=0, inference_mode=False, verbose=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            scores
        """

        # Initialization
        trainExamples = []
        fragments_nb = len(fragments)
        current_puzzle = self.game.get_init_puzzle(fragments_nb)
        current_puzzle = self.game.get_help(current_puzzle, nb_help, solution_dict)
        episodeStep = nb_help
        if verbose: print(current_puzzle)

        while True:
            if verbose: print(f"New episode: fragment {episodeStep}/{fragments_nb}.")
            episodeStep += 1
            temp = int(episodeStep < self.args['tempThreshold'])

            puz_img = self.game.pnet_input(current_puzzle, fragments)
            pi = self.mcts.getActionProb(current_puzzle, fragments, solution_dict, temp=temp)
            trainExamples.append([puz_img, pi])

            if inference_mode:
                action = np.argmax(pi)
            else:
                action = np.random.choice(len(pi), p=pi)
            if verbose: print(f"pi:{pi} and action: {action}")
            current_puzzle = self.game.get_next_state(current_puzzle, action)

            if self.game.has_game_ended(current_puzzle):
                score_f = self.game.result_fragment(current_puzzle, solution_dict)
                score_r = self.game.result_reass(current_puzzle, solution_dict)
                score_n = self.game.result_neighbors(current_puzzle, solution_dict)

                if self.args["orders"]>1:
                    puz_img = self.game.full_puzzle_img(current_puzzle, fragments)
                    x = np.array(puz_img)
                    if not self.args['wrn']:
                        x = normalize_list(x)
                    v = sigmoid(self.nnet_v.predict(x))[0][0]
                else: v = 0

                return trainExamples, score_f, score_r, score_n, v

    def execEpisodeLearn(self, fragments, solution_dict, nb_help=0, inference_mode=False, verbose=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        fragments_nb = len(fragments)
        current_puzzle = self.game.get_init_puzzle(fragments_nb)
        current_puzzle = self.game.get_help(current_puzzle, nb_help, solution_dict)
        episodeStep = nb_help
        if verbose: print(current_puzzle)
        v_baseline = [1.0]*(fragments_nb+1)
        for i in range(0,fragments_nb):
            v_baseline[i] = 0.5+0.5*i/(fragments_nb-1)
        no_miskate_made = True

        while True:
            if verbose: print("new episode. Placing fragment nb", episodeStep)

            sol_p = solution_dict[episodeStep]['position']
            episodeStep += 1
            temp = int(episodeStep < self.args['tempThreshold'])

            puz_img, frag_img = self.game.nnet_input(current_puzzle, fragments)
            pi = self.mcts.getActionProb(current_puzzle, fragments, solution_dict, temp=temp)

            action = np.argmax(pi)
            if verbose: print("pi is:", pi)
            if verbose: print("action choosen: ", action)

            sol_v_pi = v_baseline[episodeStep-1]*no_miskate_made
            sol_v = np.random.choice([0,1], p = [1-sol_v_pi, sol_v_pi])

            trainExamples.append([puz_img, frag_img, sol_p, sol_v])
            current_puzzle = self.game.get_next_state(current_puzzle, action)

            if action != sol_p:
                no_miskate_made = False

            if self.game.has_game_ended(current_puzzle):
                score_f = self.game.result_fragment(current_puzzle, solution_dict)
                score_p = self.game.result_reass(current_puzzle, solution_dict)
                return trainExamples, score_f, score_p

    def executeEpisodeGreedyP(self, fragments, solution_dict, nb_help=0, inference_mode=False, verbose=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        fragments_nb = len(fragments)
        current_puzzle = self.game.get_init_puzzle(fragments_nb)
        current_puzzle = self.game.get_help(current_puzzle, nb_help, solution_dict)
        episodeStep = nb_help
        if verbose: print(current_puzzle)
        count_p_is_sol = 0
        count_pi_is_sol = 0
        count_assertion_error = 0

        while True:
            if verbose: print("new episode. Placing fragment nb", episodeStep)
            episodeStep += 1
            temp = int(episodeStep < self.args['tempThreshold'])

            puz_img, frag_img = self.game.nnet_input(current_puzzle, fragments)
            if not self.args['wrn']:
                puz_img = normalize_list(puz_img)
                frag_img = normalize_list(frag_img)
            r = softmax(self.nnet_p.predict([puz_img, frag_img]))[0]
            pi = self.mcts.getActionProb(current_puzzle, fragments, solution_dict, temp=temp)
            rsorted = np.argsort(r)[::-1]
            loop_continue = True

            i = 0
            action2 = np.argmax(pi)
            solution = solution_dict[episodeStep-1]['position']

            while loop_continue:
                action1 = rsorted[i]
                try:
                    current_puzzle = self.game.get_next_state(current_puzzle, action1)
                except AssertionError:
                    i+=1
                    loop_continue = True
                    count_assertion_error +=1
                else:
                    loop_continue = False

            action2 = np.argmax(pi)
            solution = solution_dict[episodeStep-1]['position']
            if action1==solution:
                count_p_is_sol+=1
            if action2==solution:
                count_pi_is_sol+=1

            if self.game.has_game_ended(current_puzzle):
                score_f = self.game.result_fragment(current_puzzle, solution_dict)
                score_p = self.game.result_reass(current_puzzle, solution_dict)

                return score_f, score_p, (count_p_is_sol, count_pi_is_sol, count_assertion_error)

    def executeEpisodeGreedyV(self, fragments, solution_dict, nb_help=0, inference_mode=False, verbose=False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        fragments_nb = len(fragments)
        current_puzzle = self.game.get_init_puzzle(fragments_nb)
        current_puzzle = self.game.get_help(current_puzzle, nb_help, solution_dict)
        episodeStep = nb_help
        count_p_is_sol = 0
        if verbose: print(current_puzzle)

        while True:
            if verbose: print("new episode. Placing fragment nb", episodeStep)
            episodeStep += 1
            temp = int(episodeStep < self.args['tempThreshold'])
            v = [-1]*fragments_nb

            validmoves = [k for k, m in enumerate(self.game.get_valid_moves(current_puzzle)) if m==1]

            for i in validmoves:
                temp_current_puzzle = self.game.get_next_state(current_puzzle, i)
                try:
                    puz_img, _ = self.game.nnet_input(temp_current_puzzle, fragments)
                except AssertionError:
                    puz_img = self.game.full_puzzle_img(temp_current_puzzle, fragments)

                x = np.array(puz_img)
                if not self.args['wrn']:
                    x = normalize_list(x)
                v[i] = sigmoid(self.nnet_v.predict(x))[0][0]

            action = np.argmax(v)
            current_puzzle = self.game.get_next_state(current_puzzle, action)
            solution = solution_dict[episodeStep-1]['position']
            if action==solution:
                count_p_is_sol+=1

            if self.game.has_game_ended(current_puzzle):
                score_f = self.game.result_fragment(current_puzzle, solution_dict)
                score_p = self.game.result_reass(current_puzzle, solution_dict)
                return score_f, score_p, count_p_is_sol
