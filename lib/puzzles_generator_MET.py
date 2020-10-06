import copy
import numpy as np
import os
import random

from lib.clean_images import square_crop_resize


DEBUG = False


def standardize_fragments(images):
    images = np.array(images)
    batch_size, channels, height, width = images.shape
    images_flat = images.reshape(batch_size, channels*width*height)
    mean_values = np.mean(images_flat, axis=1, keepdims=True)
    std_values = np.std(images_flat, axis=1, keepdims=True) + 1e-5
    images_flat = (images_flat - mean_values) / std_values
    images = images_flat.reshape(images.shape)
    return images


class prepare_data_v():
    """Prepare the images per batch."""

    def __init__(self, path, phase="train", puzzle_size=80, fragment_per_side=2, fragment_size=40,  space=0, nb_helpers=2, batch_size=64, central_known=False):
        
        assert puzzle_size==fragment_per_side*fragment_size+(fragment_per_side-1)*space
        assert phase in ["train", "val", "test"]
        
        root_dir = os.path.join(path+"dataset_"+phase)
        self.paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        self.puzzle_size = puzzle_size
        self.fragment_per_side = fragment_per_side
        self.fragment_size = fragment_size
        self.space = space #between two fragments
        self.nb_helpers = nb_helpers
        self.batch_size = batch_size
        self.central_known = central_known

    def __len__(self):
        return int(np.floor(len(self.paths)/self.batch_size))

    def __getitem__(self, i):
        puzzles = np.zeros((self.batch_size, self.puzzle_size, self.puzzle_size, 3))
        solutions = np.zeros(self.batch_size, dtype=int)
        
        f_coord = [np.s_[self.fragment_size*i+self.space*(i):
                         self.fragment_size*(i+1)+self.space*(i),
                         self.fragment_size*j+self.space*(j):
                         self.fragment_size*(j+1)+self.space*(j),
                  :] for i in range(self.fragment_per_side)
                     for j in range(self.fragment_per_side)]
        
        if DEBUG:
            print(f_coord)
            print(puzzles.shape, solutions.shape)
        
        is_result_correct = False
        
        for batch_idx in range(self.batch_size):
            image = square_crop_resize(self.paths[i*self.batch_size+batch_idx], self.puzzle_size)
            fragments = [image[f]/255*2-1 for f in f_coord]            
            f_idx = list(range(len(fragments)))
            tmp = list(zip(f_coord, f_idx, fragments))
            random.shuffle(tmp)
            shuffled_coord, shuffled_idx, shuffled_frag = zip(*tmp)
            if self.central_known:
                while shuffled_idx[0] != 4:
                    random.shuffle(tmp)
                    shuffled_coord, shuffled_idx, shuffled_frag = zip(*tmp)
                    
            if DEBUG:
                print(fragments[0].shape, np.max(fragments[0]))
            
            nb_placed_fragment = np.random.randint(self.nb_helpers, self.fragment_per_side**2+1)
            is_result_correct = not is_result_correct
            
            if DEBUG:
                print(nb_placed_fragment, is_result_correct)
            
            
            if is_result_correct:
                for i in range(nb_placed_fragment):
                    puzzles[batch_idx][shuffled_coord[i]] = shuffled_frag[i]
                    
                    if DEBUG:
                        print(batch_idx, shuffled_coord[i])
                    
            else:
                mix = list(range(len(fragments)))
                while mix == list(range(len(fragments))):
                    random.shuffle(mix)
                for i in range(nb_placed_fragment):
                    puzzles[batch_idx][shuffled_coord[i]] = shuffled_frag[mix[i]]
                    if DEBUG:
                        print(batch_idx, shuffled_coord[i])
                
            solutions[batch_idx] = int(is_result_correct)
            
        return puzzles.transpose(0, 3, 1, 2).astype(dtype=np.float32), solutions.reshape(-1, 1).astype(dtype=np.float32)


class prepare_data_p():
    """Prepare the images per batch."""

    def __init__(self, path, phase="train", puzzle_size=80, fragment_per_side=2, fragment_size=40,  space=0, nb_helpers=2, batch_size=64, central_known=False):
        
        assert puzzle_size==fragment_per_side*fragment_size+(fragment_per_side-1)*space
        assert phase in ["train", "val", "test"]
        
        root_dir = os.path.join(path+"dataset_"+phase)
        self.paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        self.puzzle_size = puzzle_size
        self.fragment_per_side = fragment_per_side
        self.fragment_size = fragment_size
        self.space = space #between two fragments
        self.nb_helpers = nb_helpers
        self.batch_size = batch_size
        self.central_known = central_known
        
    def __len__(self):
        return int(np.floor(len(self.paths)/self.batch_size))

    def __getitem__(self, i):
        puzzles = np.zeros((self.batch_size, self.puzzle_size, self.puzzle_size, 3))
        next_fragments = np.zeros((self.batch_size, self.fragment_size, self.fragment_size, 3))
        solutions = np.zeros((self.batch_size, self.fragment_per_side**2), dtype=int)
        
        f_coord = [np.s_[self.fragment_size*i+self.space*(i):
                         self.fragment_size*(i+1)+self.space*(i),
                         self.fragment_size*j+self.space*(j):
                         self.fragment_size*(j+1)+self.space*(j),
                  :] for i in range(self.fragment_per_side)
                     for j in range(self.fragment_per_side)]
        
        for batch_idx in range(self.batch_size):
            image = square_crop_resize(self.paths[i*self.batch_size+batch_idx], self.puzzle_size)
            #fragments = [image[f] for f in f_coord]
            fragments = [image[f]/255*2-1 for f in f_coord]
            #fragments = standardize_fragments(fragments)
            f_idx = list(range(len(fragments)))
            tmp = list(zip(f_coord, f_idx, fragments))
            random.shuffle(tmp)
            shuffled_coord, shuffled_idx, shuffled_frag = zip(*tmp)
            if self.central_known:
                while shuffled_idx[0] != 4:
                    random.shuffle(tmp)
                    shuffled_coord, shuffled_idx, shuffled_frag = zip(*tmp)
            
            nb_placed_fragment = np.random.randint(self.nb_helpers, self.fragment_per_side**2)
            
            for j in range(nb_placed_fragment):
                puzzles[batch_idx][shuffled_coord[j]] = shuffled_frag[j]
            
            next_fragments[batch_idx] = shuffled_frag[nb_placed_fragment]
            solutions[batch_idx][shuffled_idx[nb_placed_fragment]] = 1
            
        return [puzzles.transpose(0, 3, 1, 2).astype(np.float32), next_fragments.transpose(0,3,1,2).astype(np.float32)], np.argmax(solutions, axis=1)
    
    
class prepare_fragments():
    """Prepare the images per batch."""

    def __init__(self, path, phase="train", puzzle_size=80, fragment_per_side=2, fragment_size=40,  space=0, nb_helpers=2, first_center=False):
        
        assert puzzle_size==fragment_per_side*fragment_size+(fragment_per_side-1)*space
        assert phase in ["train", "val", "test"]
        
        root_dir = os.path.join(path+"dataset_"+phase)
        self.paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        self.puzzle_size = puzzle_size
        self.fragment_per_side = fragment_per_side
        self.fragment_size = fragment_size
        self.space = space #between two fragments
        self.nb_helpers = nb_helpers
        self.first_center = first_center
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        next_fragments = np.zeros((self.fragment_per_side**2, self.fragment_size, self.fragment_size, 3))
        
        f_coord = [np.s_[self.fragment_size*i+self.space*(i):
                         self.fragment_size*(i+1)+self.space*(i),
                         self.fragment_size*j+self.space*(j):
                         self.fragment_size*(j+1)+self.space*(j),
                  :] for i in range(self.fragment_per_side)
                     for j in range(self.fragment_per_side)]
        
        image = square_crop_resize(self.paths[index], self.puzzle_size)
        fragments = [image[f]/255*2-1 for f in f_coord]
        
        return fragments