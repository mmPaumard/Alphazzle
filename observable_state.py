import numpy as np

class State():
    """Describe the current state of the game.

    Attributes:
        puzzle_size (int): Size a side of the puzzle.
        fragment_size (int): Size of a side of a fragment.
        nb_per_side (int): Number of fragments per side of puzzle.
        nb_position (int): Number of fragments in the puzzle.
        remaining_fragments_idx (list of int): List of the index of the unplaced
            fragments.
        empty_descriptor (int or NoneType): Character used to describe the
            emptiness of a puzzle cell.
        puzzle_idx (list of int or NoneType): Flat puzzle; each cell contains
            the index of the fragment that should be placed in the cell.
    """

    def __init__(self, puzzle_size, fragment_size, fragments_nb, fragments_dict=None, space=0):
        """Set up initial board configuration."""

        self.puzzle_size = puzzle_size
        self.fragment_size = fragment_size
        self.nb_per_side = int(np.sqrt(fragments_nb))
        self.nb_positions = fragments_nb
        self.space = space

        if type(fragments_dict)==type(None):
            self.fragments_dict = [{"idx":i, "position":-1} for i in range(fragments_nb)]
        else:
            self.fragments_dict = fragments_dict


    def print_state(self):
        print("fragments: ", self.fragments_dict)
        print("puzzle: ", self.get_puzzle_idx())
        print("remaining fragments:", self.get_remaining_fragments())
        print("legal_moves", self.get_empty_position())
        print("next fragment", self.get_next_fragment_idx())
        print("placed fragments", self.get_placed_fragments_idx())
        print("position, frag per side", self.nb_positions, self.nb_per_side)

    def get_puzzle_idx(self):
        puzzle = [-1]*self.nb_positions
        for f in self.fragments_dict:
            if f["position"] != -1:
                puzzle[f["position"]] = f["idx"]
        return puzzle

    def get_next_fragment_idx(self):
        if len(self.get_remaining_fragments())>0:
            return self.get_remaining_fragments()[0]
        else:
            return None

    def get_empty_position(self):
        return [k for k,i in enumerate(self.get_puzzle_idx()) if i==-1]

    def get_remaining_fragments(self):
        remaining_fragments_idx = [k for k, f in enumerate(self.fragments_dict) if f["position"] == -1]
        return remaining_fragments_idx

    def get_placed_fragments_idx(self):
        remaining_fragments_idx = [f["idx"] for f in self.fragments_dict if f["position"] != -1]
        return remaining_fragments_idx


    def is_the_game_over(self):
        """Is the game over?"""
        if self.get_empty_position() == []:
            return True
        elif len(self.get_remaining_fragments()) == 0:
            return True
        return False

    def is_the_game_won(self, solution_dict):
        if self.is_the_game_over():
            if self.score_reass(solution_dict) == 1:
                return 1
            else:
                return -1
        return 0

    def score_frag(self, solution_dict):
        """Returns the fragment-wise score of the current puzzle"""
        res = sum(i == j for i, j in zip(self.fragments_dict, solution_dict))
        return res/len(solution_dict)

    def score_reass(self, solution_dict):
        """Returns the global score of the current puzzle"""
        return int(self.fragments_dict == solution_dict)

    def score_incomplete_reass(self, solution_dict):
        res = sum(i == j if (i['position']!=-1) else 0 for i, j in zip(self.fragments_dict, solution_dict))
        size = sum(1 if i['position'] !=-1 else 0 for i in self.fragments_dict)
        # print('res: {}/{}'.format(res, size))
        return float(res==size)

    def get_neig_positions(self):
        neig_positions = {}
        for p in range(self.nb_positions):
            above = int(p-self.nb_per_side) if p>=self.nb_per_side else None
            left = p-1 if p%self.nb_per_side!=0 else None
            right = p+1 if p%self.nb_per_side!=self.nb_per_side-1 else None
            below = int(p+self.nb_per_side) if p<(self.nb_positions-self.nb_per_side) else None
            neig = [above, left, right, below]
            neig_positions[p] = [n for n in neig]
        return neig_positions

    def convert_neig_positions_to_frag(self, reass_dict, neig_pos):
        simpler_dict = {}
        simpler_reass = {i['position']:i['idx'] for i in reass_dict}
        for k,vs in zip(neig_pos.keys(), neig_pos.values()):
            simpler_dict[simpler_reass[k]] = [simpler_reass[v] if v!=None else None for v in vs]
        return simpler_dict

    def count_same(self, l1, l2):
        nb = 0
        for i, j in zip(l1, l2):
            if i==j: nb +=1
        return nb/len(l1)

    def score_neig(self, solution_dict):
        """Returns the global score of the current puzzle"""
        neig_pos = self.get_neig_positions()
        sol = self.convert_neig_positions_to_frag(solution_dict, neig_pos)
        res = self.convert_neig_positions_to_frag(self.fragments_dict, neig_pos)
        scores = [self.count_same(sol[i], res[i]) for i in sol.keys()]
        return np.average(scores)

    def get_next_fragment_img(self, fragments, verbose=False):
        """Returns the np array of the next fragment"""
        if verbose: print(self.get_next_fragment_idx())
        f = self.fragment_size
        s = self.space
        p = self.puzzle_size
        img = self.get_current_puzzle_img(fragments)
        out = np.zeros((f+s+p, f+s+p, 3), dtype=np.float32)
        out[f+s:f+s+p, f+s:f+s+p, :] = img
        idx = self.get_next_fragment_idx()
        if idx is not None:
            frag = fragments[self.get_next_fragment_idx(), ...].squeeze()
            out[s//2:s//2+f, s//2:s//2+f, :] = frag
            if verbose: print('fragment idx {}'.format(idx))
            for i in range(1, min(self.nb_per_side+1, self.nb_positions-idx)):
                if verbose: print('adding v fragment {}'.format(i))
                out[s//2+i*(f+s):s//2+i*(f+s)+f, s//2:s//2+f, :] = fragments[idx+i]
            for i in range(1, min(self.nb_per_side+1, self.nb_positions-idx-self.nb_per_side)):
                if verbose: print('adding h fragment {}'.format(i))
                out[ s//2:s//2+f,s//2+i*(f+s):s//2+i*(f+s)+f, :] = fragments[idx+self.nb_per_side+i]
        return out

    def get_current_puzzle_img(self, fragments):
        """Returns the np array of the puzzle"""
        puzzle = np.zeros((self.puzzle_size, self.puzzle_size, 3), dtype=np.float32)
        try:
            for k, idx in enumerate(self.get_puzzle_idx()):
                if idx!=-1:
                    y_min = int((self.fragment_size+self.space)*(k//self.nb_per_side))
                    y_max = int(y_min + self.fragment_size)
                    x_min = int((self.fragment_size+self.space)*(k%self.nb_per_side))
                    x_max = int(x_min + self.fragment_size)
                    puzzle[y_min+self.space//2:y_max+self.space//2, x_min+self.space//2:x_max+self.space//2,:] = fragments[idx]
        except TypeError:
            print(y_min, y_max, x_min, x_max)
            print(puzzle.shape)
            self.print_state()
            print(self.fragment_size,self.space,self.nb_per_side)
            raise TypeError
        return puzzle

    def execute_move(self, move):
        assert(move in self.get_empty_position())
        self.fragments_dict[self.get_next_fragment_idx()]["position"] = move
