import math
import numpy as np
import torch

from lib.utils import sigmoid, softmax
EPS = 1000 #1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet_p, nnet_v, args):
        self.game = game
        self.nnet_p = nnet_p
        self.nnet_v = nnet_v
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.has_game_ended ended for board s
        self.Vs = {}        # stores game.get_valid_moves for board s
        self.Vt = {}        # stores v for terminal

    def print_state(self):
        print("QSA", self.Qsa)
        print("NSA", self.Nsa)
        print("Ns", self.Ns)
        print("Ps", self.Ps)
        print("Es", self.Es)
        print("Vs", self.Vs)

    def print_specific_state(self, puzzle, action):
        print("QSA", self.Qsa[(str(puzzle), action)])
        print("NSA", self.Nsa[(str(puzzle), action)])
        print("Ns", self.Ns[str(puzzle)])
        print("Ps", self.Ps[str(puzzle)])
        print("Es", self.Es)
        print("Vs", self.Vs)

    def getActionProb(self, current_puzzle, fragments, solution_dict, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args['numMCTSSims']):
            #print("--------------------new sim---------------------")
            self.search(current_puzzle, fragments, solution_dict)
            print('\rsim {}/{}'.format(i, self.args['numMCTSSims']), end='')
        print('')
        #print("done")

        s = self.game.string_representation(current_puzzle)
        if self.args['useQSA']:
            counts = [self.Qsa[(s,a)] if (s,a) in self.Qsa else 0 for a in range(self.game.action_size)]
        else:
            counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.action_size)]

        print('counts: {}'.format(counts))

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]

        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, current_puzzle, fragments, solution_dict, verbose=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard → now positive
        """

        fragments_nb = len(fragments)
        s = self.game.string_representation(current_puzzle)

        if s not in self.Es:
            if verbose: print("if s has not been visited, create s in ES")
            self.Es[s] = self.game.has_game_ended(current_puzzle)
        if self.Es[s]:
            if verbose: print("terminal node, puzzle:", current_puzzle)

            if s not in self.Vt:
                # nn_puzzle = self.game.full_puzzle_img(current_puzzle, fragments)
                # if self.args['disable_v1']==1:
                #     v = 1
                if self.args['disable_v2']==1: #ground truth, puzzle-wise
                    if verbose: print('using ground truth')
                    v = self.game.result_fragment(current_puzzle, solution_dict)
                    print('   terminal node comparing gt: {:.3f}'.format(v), end='')
                else:
                    if verbose: print('predicting v using nnet')
                    nn_puzzle = self.game.vnet_input(current_puzzle, fragments)
                    # v = (self.nnet_v(nn_puzzle).detach().cpu().numpy()[0])
                    v = self.nnet_v(nn_puzzle).detach().cpu().numpy()[0]
                    v_gt = self.game.result_fragment(current_puzzle, solution_dict)
                    print('        terminal node comparing gt: {:.3f} to pred {}'.format(v_gt, v), end='')
                self.Vt[s] = v
                if verbose: print(v)
            return self.Vt[s]
            #return self.Es[s]

        if s not in self.Ps:
            if verbose: print("leaf node, puzzle:", current_puzzle)
            nn_puzzle = self.game.pnet_input(current_puzzle, fragments)
            # import matplotlib.pyplot as plt
            # img = nn_puzzle.cpu().numpy().squeeze()
            # img = img.transpose(1, 2, 0)
            # img = (img - np.min(img)) / np.ptp(img)
            # plt.clf()
            # plt.imshow(img)
            if self.args['disable_p']:
                self.Ps[s] = np.ones((self.args['position_nb']))
            else:
                x1 = nn_puzzle
                nb_f = int(np.sqrt(self.game.action_size))
                mask = (1-self.game.get_valid_moves(current_puzzle))
                mask = mask.reshape(1, nb_f, nb_f)
                mask = np.pad(mask, [(0,0), (1,0), (1,0)], mode='constant', constant_values=1.)
                mask = torch.tensor(mask).bool().cuda()
                self.Ps[s] = softmax(self.nnet_p(x1, mask).detach().cpu().numpy()[0])
            if verbose: print(self.Ps[s])

            valids = self.game.get_valid_moves(current_puzzle)
            self.Ps[s] = self.Ps[s]*valids      # set invalid moves to 0

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                #print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # P_str = '['
            # for pi in self.Ps[s]: P_str += ' {:.2f}'.format(pi)
            # P_str += ']'
            # print('   policy P: {}'.format(P_str), end='')

            self.Vs[s] = valids
            self.Ns[s] = 0

            if self.args['disable_v1']==1:
                v = self.game.result_fragment(current_puzzle, solution_dict)
            # elif self.args['disable_v2']==1: #ground truth, puzzle-wise
                print('   normal node comparing gt: {:0.3f}'.format(v), end='')
            else:
                nn_puzzle = self.game.vnet_input(current_puzzle, fragments)
                nb_f = int(np.sqrt(self.game.action_size))
                mask = (1-self.game.get_valid_moves(current_puzzle))
                mask = torch.tensor(mask.reshape(1, nb_f, nb_f)).bool().cuda()
                # mask = None
                v = self.nnet_v(nn_puzzle, mask).detach().cpu().numpy()[0]
                v_gt = self.game.result_fragment(current_puzzle, solution_dict)
                # print('   normal node comparing gt: {:0.3f} to pred {}'.format(v_gt, v), end='')
                # print("V IS: ", v)

            # plt.title('V : {}'.format(v))
            # plt.pause(0.1)
            return v

        #print('explored node')

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.action_size):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['lambda']*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])*self.Ps[s][a]
                else:
                    u = self.args['lambda']*math.sqrt(self.Ns[s] + EPS)*self.Ps[s][a]     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(current_puzzle, a)

        v = self.search(next_s, fragments, solution_dict)                                                       # recursion

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        #print("action ", a)
        return v
