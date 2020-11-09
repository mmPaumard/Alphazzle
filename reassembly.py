import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn

from datetime import date
from PIL import Image
from poutyne.framework import Model ##replace poutyne @TODO
from lib.vit_pytorch import ViT
import pytorch_lightning as pl

from torch.optim import Adam

from game import Game
from coach import Coach
from lib.nn_architectures import ModelP
from lib.puzzles_generator_MET import prepare_data_p, prepare_data_v, prepare_fragments
from lib.wide_resnet_torch import create_model

np.set_printoptions(precision=2)


################################################################################

def prepare_nnets(args):
    """Initializes the neural networks.

    Parameters:
        args(dict):         current setting

    Returns:
        torchvision.models: P and V neural networks, with their weights
    """

    DATASET_PATH=args['dir_global']
    PUZZLE_SIZE=args['puzzle_size']
    NB_FRAG = args['fragment_per_side']
    FRAG_SIZE = args['fragment_size']
    SPACE_SIZE = args['space_size']
    CONV_HEAD = args['conv_head']

    if VERBOSE: print('===== PREPARE NNETS =====')
    img_size = (NB_FRAG+1)*(FRAG_SIZE+SPACE_SIZE)
    frg_size = FRAG_SIZE+SPACE_SIZE

    print(img_size, frg_size, SPACE_SIZE)
    ### Prepare neural network P ####

    # lightning wrapper
    class LitModelP(pl.LightningModule):

        def __init__(self):
            super().__init__()
            self.vit = ViT(image_size=img_size,
                           patch_size=frg_size,
                           num_classes=NB_FRAG**2,
                           dim=1024,
                           depth=4,
                           heads=8,
                           mlp_dim=2048,
                           conv_head=CONV_HEAD)
            self.accuracy = pl.metrics.Accuracy()

            self.example_input_array = torch.randn((1, 3, img_size, img_size))

        def forward(self, x):
            # in lightning, forward defines the prediction/inference actions
            out = self.vit(x)
            return out

        def training_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            y_hat = self.vit(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            acc = self.accuracy(y_hat, y)
            # Logging to TensorBoard by default
            self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
            self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            y_hat = self.vit(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            acc = self.accuracy(y_hat, y)
            # Logging to TensorBoard by default
            self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True)
            self.log('val_acc', acc, on_epoch=True, on_step=False, logger=True)
            return loss


        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
            return optimizer

    if args['p_weight_path']:
        model_p = LitModelP.load_from_checkpoint(args['p_weight_path'])
    else:
        model_p = LitModelP()

    ## V

    img_size = (NB_FRAG)*(FRAG_SIZE+SPACE_SIZE)

    # lightning wrapper
    class LitModelV(pl.LightningModule):

        def __init__(self):
            super().__init__()
            self.vit = ViT(image_size=img_size,
                           patch_size=frg_size,
                           num_classes=1,
                           dim=1024,
                           depth=4,
                           heads=8,
                           mlp_dim=2048,
                           conv_head=CONV_HEAD)
            self.accuracy = pl.metrics.Accuracy()

            self.example_input_array = torch.randn((1, 3, img_size, img_size))

        def forward(self, x):
            # in lightning, forward defines the prediction/inference actions
            out = self.vit(x)
            return out

        def training_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            y_hat = self.vit(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
            acc = self.accuracy(y_hat, y)
            # Logging to TensorBoard by default
            self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
            self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            # training_step defined the train loop.
            # It is independent of forward
            x, y = batch
            y_hat = self.vit(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
            acc = self.accuracy(y_hat, y)
            # Logging to TensorBoard by default
            self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True)
            self.log('val_acc', acc, on_epoch=True, on_step=False, logger=True)
            return loss


        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
            return optimizer

    if args['v_weight_path']:
        model_v = LitModelV.load_from_checkpoint(args['v_weight_path'])
    else:
        model_v = LitModelV()


    if VERBOSE: print("weights loaded")
    return model_p.cuda(), model_v.cuda()

################################################################################

def prepare_problem(args, g, current_img_index, phase="val", middle=False):
    """For a puzzle, returns the fragments and the solution.

    Parameters:
        args(dict):             the current setting
        g:                      the game
        current_img_index(int): the puzzle index
        phase(str):             the dataset we use
        middle(bool):           True if we place the middle fragment first

    Returns:
        np.array:       the shuffled array of fragments
        dict:           the dictionnary of solutions
    """
    fragments = prepare_fragments(path=args['dir_global'], phase=phase,
                                  puzzle_size=args['puzzle_size'], fragment_per_side=args['fragment_per_side'],
                                  fragment_size=args['fragment_size'], space=args['space_size'])[current_img_index]
    fragments_idx = list(range(len(fragments)))

    if middle:
        fragment_middle = np.array(fragments[4])
        fragments_0 = fragments[:4]
        fragments_1 = fragments[5:]
        fragments = np.concatenate((fragments_0, fragments_1))
        fragments_idx = list(range(len(fragments)+1))
        fragments_idx.pop(4)

    shuffled = list(zip(fragments, fragments_idx))
    random.shuffle(shuffled)
    fragments, fragments_idx = zip(*shuffled)
    fragments = np.array(fragments)

    if middle:
        fragments = np.concatenate(([fragment_middle], list(fragments)))
        fragments_idx = [4].extend(fragments_idx)

    solution_dict = g.get_init_puzzle(args['fragments_nb'])
    for i in range(len(solution_dict)):
        solution_dict[i]['position'] = fragments_idx[i]

    return fragments, solution_dict

################################################################################

def make_reassemblies(args, model_p, model_v, nb_img=None):
    """Builds the reassemblies

    Parameters:
        args(dict):             the current setting
        model_p:                the neural network P
        model_v:                the neural network V
        nb_img(int):            the number of reassemblies to do

    Returns:
        tuple:       scores
    """

    if VERBOSE: print("===== EVALUATE PUZZ =====")
    # get the total number of puzzles
    if nb_img==None: nb_img = len(os.listdir(args['dir_valid']))

    score_reas = [0,0,0]
    score_frag = [0,0,0]
    score_neig = [0,0,0]

    print(f"Each score presents the result for the best, first and worst predicted V, for {args['orders']} random orders of fragments seen")

    for current_img_id in range(nb_img):
        predicted_vs = [0.0]*args['orders']
        scores_reas = [0.0]*args['orders']
        scores_frag = [0.0]*args['orders']
        scores_neig = [0.0]*args['orders']

        for q in range(args['orders']):
            g = Game(args['puzzle_size'], args['fragment_size'],
                     args['fragments_nb'], space=args['space_size'])
            c = Coach(g, model_p, model_v, args)

            fragments, solution_dict = prepare_problem(args, g, current_img_id,
                                                       middle=args['central_frag'])
            scores = c.executeEpisode(fragments, solution_dict,
                                      nb_help=args['numHelp'],
                                      inference_mode=args['inference'],
                                      verbose=False)
            _,f,r,n,v = scores
            frag_ok_minus_help = int(f*args['fragments_nb']) - args['numHelp']
            f = frag_ok_minus_help/(args['fragments_nb']-args['numHelp'])

            scores_frag[q] = f
            scores_reas[q] = r
            scores_neig[q] = n
            predicted_vs[q] = v

        argv = np.argmax(predicted_vs)
        score_frag[0] += scores_frag[argv]
        score_reas[0] += scores_reas[argv]
        score_neig[0] += scores_neig[argv]
        score_frag[1] += scores_frag[0]
        score_reas[1] += scores_reas[0]
        score_neig[1] += scores_neig[0]
        argv = np.argmin(predicted_vs)
        score_frag[2] += scores_frag[argv]
        score_reas[2] += scores_reas[argv]
        score_neig[2] += scores_neig[argv]

        print(f"Puzzle {current_img_id}: fragments scores are {[round(score_frag[i]/(current_img_id+1)*100,2) for i in range(3)]}% and we count {score_reas} correct reassemblies. Neighbors scores are {[round(s*100/(current_img_id+1),2) for s in score_neig]}.")

    return score_frag, score_reas

################################################################################

def learn(args, model_p, model_v):
    print("numSim is", args['numMCTSSims'])

    nb_img = len(os.listdir(args['dir_valid']))
    k = 0

    for iters in range(args['numIters']):
        if VERBOSE: print("===== MAKE PUZZLES =====", iters)
        scoref_final_t = 0
        scorep_final_t = 0
        trs = []
        tr_score = []
        tr_puzzles = []
        tr_fragments = []
        tr_solutions_p = []
        tr_solutions_v = []

        for current_img_id in range(iters*args['size_batch_train'], iters*args['size_batch_train']+args['size_batch_train']):
            g = Game(args['puzzle_size'], args['fragment_size'], args['fragments_nb'], space=args['space_size'])
            c = Coach(g, model_p, model_v, args)

            fragments, solution_dict = prepare_problem(args, g, current_img_id%nb_img, phase="train")

            tr, sf, sp = c.execEpisodeLearn(fragments, solution_dict, nb_help=args['numHelp'],
                                        inference_mode=args['inference'], verbose=False)

            total_frag_ok = int(sf*args['fragments_nb'])
            frag_ok_minus_help = total_frag_ok - args['numHelp']
            sf = frag_ok_minus_help/(args['fragments_nb']-args['numHelp'])

            scoref_final_t += sf
            scorep_final_t += sp
            trs.append(tr)

        print("train on ", nb_img, '|', np.round(scoref_final_t/100, 2), '|', round(scorep_final_t, 2))

        for i in trs:
            for j in i:
                tr_puzzles.append(j[0][0])
                tr_fragments.append(j[1][0])
                tr_solutions_p.append(j[2])
                tr_solutions_v.append([float(j[3])])

        p = np.asarray(tr_puzzles)
        f = np.asarray(tr_fragments)
        sp = np.asarray(tr_solutions_p)
        sv = np.asarray(tr_solutions_v)

        save_dir_name = "llmodels/"+date.today().strftime('%Y%b%d')+"/"
        if not os.path.exists(save_dir_name):
            os.mkdir(save_dir_name)
        filepath_v = save_dir_name+"v_"+args['struct']+"_e"+str(iters)+".h5"
        filepath_p = save_dir_name+"p_"+args['struct']+"_e"+str(iters)+".h5"

        if VERBOSE: print("===== TRAIN =====")
        if args['trainP']:
            model_p.fit([p,f], sp, epochs=1)
            model_p.save_weights(filepath_p.format(epoch=iters))
        if args['trainV']:
            model_v.fit(p, sv, epochs=1)
            model_v.save_weights(filepath_v.format(epoch=iters))

        if VERBOSE: print("==== EVAL ====")
        scores = make_reassemblies(args, model_p, model_v, args['size_batch_val'])
        return scores


################################################################################

def main(args):
    model_p, model_v = prepare_nnets(args)
    if args['learn']:
        scores = learn(args, model_p, model_v)

    scores = make_reassemblies(args, model_p, model_v)
    return scores

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--verb", nargs=1)
    parser.add_argument("-g", "--orders", nargs=1)
    parser.add_argument("-q", "--qsa", nargs=1)
    parser.add_argument("-w", "--conv_head", action="store_true", default=False)

    parser.add_argument("-a", "--learn", nargs=1)
    parser.add_argument("-u", "--learnV", nargs=1)
    parser.add_argument("-b", "--sb_train", nargs=1)
    parser.add_argument("-d", "--sb_val", nargs=1)

    parser.add_argument("-l", "--lambd", nargs=1)
    parser.add_argument("-r", "--nb_help", nargs=1)
    parser.add_argument("-m", "--nb_mcts", nargs=1)
    parser.add_argument("-t", "--temp", nargs=1)
    parser.add_argument("-c", "--central_fragment", nargs=1)

    parser.add_argument("-e", "--nb_epochs", nargs=1)
    parser.add_argument("-i", "--nb_iters", nargs=1)

    parser.add_argument("-f", "--frag_size", nargs=1)
    parser.add_argument("-s", "--space_size", nargs=1)
    parser.add_argument("-n", "--nb_frag_per_side", nargs=1)

    parser.add_argument("-p", "--p_weight", nargs=1)
    parser.add_argument("-v", "--v_weight", nargs=1)

    parser.add_argument("-j", "--inference", nargs=1)
    parser.add_argument("-x", "--predict_p", nargs=1)
    parser.add_argument("-y", "--predict_v1", nargs=1)
    parser.add_argument("-z", "--predict_v2", nargs=1)
    #parser.add_argument("-k", "--empty", nargs=1)

    parsed_args = parser.parse_args()

    global VERBOSE, ORDERS, QSA, WRN
    VERBOSE = int(parsed_args.verb[0]) if parsed_args.verb else 1
    ORDERS = int(parsed_args.orders[0]) if parsed_args.orders else 1
    QSA = int(parsed_args.qsa[0]) if parsed_args.qsa else 0
    CONV_HEAD = parsed_args.conv_head

    global LEARN, LEARNV, SBTRAIN, SBVAL
    LEARN = int(parsed_args.learn[0]) if parsed_args.learn else 0
    LEARNV = int(parsed_args.learnV[0]) if parsed_args.learnV else LEARN
    SBTRAIN = int(parsed_args.sb_train[0]) if parsed_args.sb_train else 100
    SBVAL = int(parsed_args.sb_val[0]) if parsed_args.sb_val else 2000

    global LAMBDA, NB_HELP, NB_MCTS, TEMP, CENTRAL_FRAGMENT
    LAMBDA = float(parsed_args.lambd[0]) if parsed_args.lambd else 1
    NB_HELP = int(parsed_args.nb_help[0]) if parsed_args.nb_help else 0
    NB_MCTS = int(parsed_args.nb_mcts[0]) if parsed_args.nb_mcts else 1000
    TEMP = float(parsed_args.temp[0]) if parsed_args.temp else 15
    CENTRAL_FRAGMENT = bool(int(parsed_args.central_fragment[0])) if parsed_args.central_fragment else False

    global NB_EPOCHS, NB_ITERS
    NB_EPOCHS = int(parsed_args.nb_epochs[0]) if parsed_args.nb_epochs else 100
    NB_ITERS = int(parsed_args.nb_iters[0]) if parsed_args.nb_iters else 1000

    global FRAG_SIZE, SPACE_SIZE, NB_FRAG_PER_SIDE
    FRAG_SIZE = int(parsed_args.frag_size[0]) if parsed_args.frag_size else 40
    SPACE_SIZE = int(parsed_args.space_size[0]) if parsed_args.space_size else 4
    NB_FRAG_PER_SIDE = int(parsed_args.nb_frag_per_side[0]) if parsed_args.nb_frag_per_side else 3

    global STRUCT, P_WEIGHT, V_WEIGHT
    STRUCT = str(FRAG_SIZE)+("-"+str(SPACE_SIZE)+"-"+str(FRAG_SIZE))*(NB_FRAG_PER_SIDE-1)+"_"
    p_weight = parsed_args.p_weight[0] if parsed_args.p_weight else None
    v_weight = parsed_args.v_weight[0] if parsed_args.v_weight else None
    P_WEIGHT = p_weight
    V_WEIGHT = v_weight

    global DISABLE_P, DISABLE_V1, DISABLE_V2, INFERENCE
    INFERENCE = bool(int(parsed_args.inference[0])) if parsed_args.inference else True
    DISABLE_P = bool(int(parsed_args.predict_p[0])) if parsed_args.predict_p else False
    DISABLE_V1 = int(parsed_args.predict_v1[0]) if parsed_args.predict_v1 else 0
    DISABLE_V2 = int(parsed_args.predict_v2[0]) if parsed_args.predict_v2 else 0

    # assert P_WEIGHT[15:] in os.listdir('saved_models')
    # assert V_WEIGHT[15:] in os.listdir('saved_models')

    print('verbose, nb_order, useQSA, useWRN:', VERBOSE, ORDERS, QSA, CONV_HEAD)
    print('finetuning (learn, learnVtoo, batchtrain, batchval):', LEARN, LEARNV, SBTRAIN, SBVAL)
    print('parameters (lambda, help, mcts, temp, centralf):', LAMBDA, NB_HELP, NB_MCTS, TEMP, CENTRAL_FRAGMENT)
    print('epochs and iters:', NB_EPOCHS, NB_ITERS)
    print('puzzle shape (f,s,qt):', FRAG_SIZE, SPACE_SIZE, NB_FRAG_PER_SIDE)
    print('nnets:', P_WEIGHT, V_WEIGHT)
    print('inference, disable P/V1/V2:', DISABLE_P, DISABLE_V1, DISABLE_V2, INFERENCE)

    args = {
        'lambda': LAMBDA, #cpuct
        'numHelp': NB_HELP,
        'numMCTSSims': NB_MCTS,
        'tempThreshold': TEMP,
        'central_frag': CENTRAL_FRAGMENT,

        'numIters': NB_ITERS,
        'numEps': NB_EPOCHS,
        'maxlenOfQueue': 200000,
        'numItersForTrainExamplesHistory': 20,

        'fragment_size': FRAG_SIZE,
        'space_size': SPACE_SIZE,
        'fragment_per_side': NB_FRAG_PER_SIDE,
        'puzzle_size': FRAG_SIZE*NB_FRAG_PER_SIDE+SPACE_SIZE*(NB_FRAG_PER_SIDE),
        'fragments_nb': NB_FRAG_PER_SIDE**2,
        'position_nb': NB_FRAG_PER_SIDE**2,

        'dir_train': '/home/david/Bases/fp_puzzles/dataset_train/',
        'dir_valid': '/home/david/Bases/fp_puzzles/dataset_val/',
        'dir_global': '/home/david/Bases/fp_puzzles/',
        'checkpoint': './temp/',
        'p_weight_path': P_WEIGHT,
        'v_weight_path': V_WEIGHT,

        'inference': True,
        'disable_p': DISABLE_P,
        'disable_v1': DISABLE_V1,
        'disable_v2': DISABLE_V2,

        'learn': LEARN,
        'trainP': True,
        'trainV': bool(LEARNV),
        'size_batch_train': SBTRAIN,
        'size_batch_val': SBVAL,

        'orders': ORDERS,
        'useQSA': QSA,
        'struct': STRUCT,
        'conv_head': CONV_HEAD
    }

    main(args)
