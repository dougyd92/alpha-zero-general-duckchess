import logging
import math
import numpy as np

from collections import defaultdict

EPS = 1e-8

log = logging.getLogger(__name__)

class TreeLevel():
    """
    Holds all the nodes at a certain tree depth.
    This is so higher levels can be discarded as the game progresses.
    """
    def __init__(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.nodes = defaultdict(TreeLevel)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        depth = canonicalBoard.move_count # use to prune unneeded nodes in the tree

        counts = [self.nodes[depth].Nsa[(s, a)] if (s, a) in self.nodes[depth].Nsa else 0 for a in range(self.game.getActionSize())]

        if (depth-1) in self.nodes:
            del self.nodes[depth-1] # Discard the parts of the tree that won't be used anymore

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)
        depth = canonicalBoard.move_count

        if s not in self.nodes[depth].Es:
            self.nodes[depth].Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.nodes[depth].Es[s] != 0:
            # terminal node
            return -self.nodes[depth].Es[s]

        if s not in self.nodes[depth].Ps:
            # leaf node
            self.nodes[depth].Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.nodes[depth].Ps[s])
            if sum_Ps_s > 0:
                self.nodes[depth].Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] + valids
                self.nodes[depth].Ps[s] /= np.sum(self.nodes[depth].Ps[s])

            self.nodes[depth].Vs[s] = valids
            self.nodes[depth].Ns[s] = 0
            return -v

        valids = self.nodes[depth].Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.nodes[depth].Qsa:
                    u = self.nodes[depth].Qsa[(s, a)] + self.args.cpuct * self.nodes[depth].Ps[s][a] * math.sqrt(self.nodes[depth].Ns[s]) / (
                            1 + self.nodes[depth].Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.nodes[depth].Ps[s][a] * math.sqrt(self.nodes[depth].Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.nodes[depth].Qsa:
            self.nodes[depth].Qsa[(s, a)] = (self.nodes[depth].Nsa[(s, a)] * self.nodes[depth].Qsa[(s, a)] + v) / (self.nodes[depth].Nsa[(s, a)] + 1)
            self.nodes[depth].Nsa[(s, a)] += 1

        else:
            self.nodes[depth].Qsa[(s, a)] = v
            self.nodes[depth].Nsa[(s, a)] = 1

        self.nodes[depth].Ns[s] += 1
        return -v
