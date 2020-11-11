import copy
import numpy as np
import torch

from observable_state import State

class Game():
    """
    This class specifies the rules of the puzzle solving.

    A game is characterized by the size of the puzzle and the size of the
    fragment, in pixel. Both of them are squares.

    Attributes:
        puzzle_size (int): The side size of the puzzle in pixels.
        fragment_size (int): The side size of a fragment in pixels.
        action_size (int): The number of actions that exists.

    Note:
        Currently, the action "do not use this fragment" is not implemented. It
        implies that the number of actions is the number of positions.

    The games rules contains all the methods that can be operated on any state:
    e.g., clear the game, obtain the next fragment, place it and evaluate the
    score of the game (if applicable). It forbids to place two fragments at the
    same position and require all the position to be filled to end the game.

    Any state can be described by the number of fragments and the ordered list
    of fragments indices, representing the current puzzle state.
    E.g., [-1, -1, 2, 0] is a 2×2 puzzle were the first row is empty; the second
    row contains the fragment 2 and the fragment 0.
    """

    def __init__(self, puzzle_size, fragment_size, fragments_nb, space=0):
        self.puzzle_size = puzzle_size
        self.fragment_size = fragment_size
        self.space = space
        self.action_size = fragments_nb


    def get_init_puzzle(self, fragments_nb):
        """Reset the puzzle.

        Args:
            fragments_nb (int): The number of fragments to place.

        Returns:
            list of int: The current representation of the puzzle.

        Note:
            The cleared puzzle must be [-1, -1, -1, ...]. Its length should be
            equal to the number of positions.

        Note:
            This method is used by Coach.executeEpisode.
        """
        s = State(self.puzzle_size, self.fragment_size, fragments_nb)
        assert len(s.get_puzzle_idx()) == s.nb_positions, "check the puzzle lenght"
        assert set(s.get_puzzle_idx()) == {-1}, "verify that the puzzle is empty"
        return s.fragments_dict


    def get_help(self, current_fdict, help_nb, solution_dict, verbose=False):
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        if verbose: s.print_state()
        assert(not s.is_the_game_over())
        for i in range(help_nb):
            idx = s.get_next_fragment_idx()
            s.execute_move(solution_dict[idx]["position"])
        return s.fragments_dict


    def get_valid_moves(self, current_fdict):
        """Evaluate which moves are valid.

        Returns:
            np.array of int: a binary vector of length self.action_size, with 1 for
            the moves that are valid for the current puzzle, 0 for invalid moves

        Note:
            This method is used by the MCTS.search: it stores the valid moves for each
            states.
        """
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        empty_pos = s.get_empty_position()
        try:
            legalMoves = [0]*self.action_size
            for p in empty_pos:
                legalMoves[p] = 1
        except IndexError:
            print(legalMoves)
            print(empty_pos)
            print(current_fdict)
            s.print_state()
            raise IndexError
        return np.array(legalMoves)


    def get_next_state(self, current_fdict, move):
        """ Apply a move to the current puzzle.

        Args:
            fragments_nb (int): The number of fragments to place.
            current_puzzle (list of int): The current representation of the puzzle.
            move (int): The position where the next fragment is to be placed.

        Returns:
            list of int: The next representation of the puzzle.

        Note:
            This method is used by Coach.executeEpisode and by the MCTS.search.
        """
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        #assert move<self.action_size, "move not in action size"
        #assert self.get_valid_moves(s.get_puzzle_idx())[move]==1, "check that the move is valid"
        s.execute_move(move)
        return s.fragments_dict


    def has_game_ended(self, current_fdict):
        """Evaluates if the game has ended.

        Args:
            fragments_nb (int): The number of fragments to place.
            current_puzzle (list of int): The current representation of the puzzle.

        Returns:
            bool: True if the game has ended.
        """
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        return s.is_the_game_over()


    def result_fragment(self, current_fdict, solution_dict):
        """Compute the score of the completed puzzle, fragment-wise.

        Args:
            fragments_nb (int): The number of fragments to place.
            current_puzzle (list of int): The current representation of the puzzle.
            solution_idx (list of int): The representation of the solution.

        Returns:
            float: The score of the reassembly. 1 means the reassembly is perfect.
        """
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        return s.score_frag(solution_dict)


    def result_reass(self, current_fdict, solution_dict):
        """Compute the score of the completed puzzle."""
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        return s.score_reass(solution_dict)


    def result_neighbors(self, current_fdict, solution_dict):
        """Compute the score of the completed puzzle"""
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        return s.score_neig(solution_dict)

    def string_representation(self, current_fdict):
        """The string representation is used by the MCTS tree (a dictionnary)."""
        #return np.array(current_puzzle).tostring()
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict))
        puzzle = s.get_puzzle_idx()
        return str(puzzle)


    def pnet_input(self, current_fdict, fragments, verbose=False):
        """Returns the np array of the puzzle"""
        fragments_nb = len(fragments)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict), space=self.space)
        if verbose: print(s.get_remaining_fragments(), current_fdict)
        # assert(s.get_remaining_fragments() != [])

        nnet_fragment = s.get_next_fragment_img(fragments).transpose(2,0,1)

        return torch.tensor(nnet_fragment.reshape(1, 3, self.puzzle_size+self.space+self.fragment_size, self.puzzle_size+self.space+self.fragment_size)).cuda()
        # return nnet_puzzle.astype(np.float32).reshape(1,3,self.puzzle_size,self.puzzle_size), nnet_fragment.astype(np.float32).reshape(1,3,self.fragment_size,self.fragment_size)

    def vnet_input(self, current_fdict, fragments, verbose=False):
        """Returns the np array of the puzzle"""
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict), space=self.space)
        if verbose: print(s.get_puzzle_idx())

        nnet_puzzle = s.get_current_puzzle_img(fragments).transpose(2,0,1)
        nnet_puzzle = nnet_puzzle.reshape(1, 3, self.puzzle_size, self.puzzle_size)

        return torch.tensor(nnet_puzzle).cuda()


    def full_puzzle_img(self, current_fdict, fragments):
        """Returns the np array of the puzzle"""
        fragments_nb = len(current_fdict)
        s = State(self.puzzle_size, self.fragment_size, fragments_nb, copy.deepcopy(current_fdict), space=self.space)
        nnet_puzzle = s.get_current_puzzle_img(fragments).transpose(2,0,1)
        return nnet_puzzle.astype(np.float32).reshape(1,3,self.puzzle_size,self.puzzle_size)
