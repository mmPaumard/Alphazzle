import argparse
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models

from datetime import date
from poutyne.framework import Model ##replace poutyne @TODO
from torch.optim import Adam
from torchvision import transforms

from lib.nn_architectures import ModelP, ModelPPretrained
from lib.puzzles_generator_MET import prepare_data_p, prepare_data_v
from lib.utils import Flatten, SaveModel
from lib.wide_resnet_torch import create_model


# If options_for_defining_WRN_28_4["pool"] is "none" then the output of the
# network is a feature map. If options_for_defining_WRN_28_4["pool"] is
# "avg" then a global average pooling layer is added on top of the last
# layer of network and the output of the network is now a feature vector.
options_for_defining_WRN_28_4 = {
    "depth": 28,
    "widen_factor": 4,
    "pool": "none"
}

def normalize_list(x):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    x = [normalize(torch.tensor(i)) for i in x]
    x = torch.stack(x)
    return x

def main():
    print("loading datasets")
    ##### Prepare data for P ########
    dataset_train_p = prepare_data_p(path=DATASET_PATH, phase="train", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT)
    dataset_valid_p = prepare_data_p(path=DATASET_PATH, phase="val", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT)
    print("p dataset has been loaded")

    ##### Prepare data for V ########
    dataset_train_v = prepare_data_v(path=DATASET_PATH, phase="train", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT)
    dataset_valid_v = prepare_data_v(path=DATASET_PATH, phase="val", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT)
    print("v dataset has been loaded")

    ## Prepare the neural networks ##
    if WRN:
        WRN_28_4 = create_model(options_for_defining_WRN_28_4)
        feature_extractor = nn.Sequential(WRN_28_4,nn.MaxPool2d(2),nn.Flatten())
    else:
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity() # removing classification layer
        feature_extractor = resnet18

    x,_ = dataset_train_p[0]
    x1, x2 = x
    if WRN: x1, x2 = torch.tensor(x1), torch.tensor(x2)
    else: x1, x2 = normalize_list(x1), normalize_list(x2)
    v1, v2 = feature_extractor(x1), feature_extractor(x2)
    img_size, frg_size = v1.shape[1], v2.shape[1]

    ### Prepare neural network P ####
    if WRN:
        model = ModelP(wrn=WRN_28_4, nb_out=NB_FRAG**2, img_size=img_size, frg_size=frg_size)
    else:
        model = ModelPPretrained(base=feature_extractor, nb_out=NB_FRAG**2, img_size=img_size, frg_size=frg_size)
    adam = Adam(model.parameters(), lr=0.001)
    model_p = Model(model, adam, 'crossentropy', batch_metrics=['accuracy'])
    model_p.cuda()

    ### Prepare neural network V ####
    if WRN:
        model = nn.Sequential(feature_extractor,
                              nn.Linear(img_size, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 1))
    else:
        model = nn.Sequential(feature_extractor,
                              nn.Linear(img_size, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 1))

    adam = Adam(model.parameters(), lr=0.001)
    model_v = Model(model, adam, 'BCEWithLogits', batch_metrics=['bin_acc'])
    model_v.cuda()

    ####### Prepare the save ########
    save_dir_name = "models/"+date.today().strftime('%Y%b%d')+"/"
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
    filepath_w = save_dir_name+"pv_"+STRUCT+"_e{epoch:02d}_{val_acc:.2f}.h5"
    filepath_v = save_dir_name+"v_"+STRUCT+"_e{epoch:02d}_{val_acc:.2f}.h5"
    filepath_p = save_dir_name+"p_"+STRUCT+"_e{epoch:02d}_{val_acc:.2f}.h5"

    ########### LEARNING ############
    print("ready for learning")
    bestP, bestV = 0, 0
    for e in range(NB_EPOCHS):
        print('epoch {}'.format(e+1))
        n = len(dataset_train_p)
        r = np.random.permutation(n)
        lp = ap = lv = av = t = 0
        for i in r:
            t += 1
            x, y = dataset_train_p[i]
            if not WRN: x = [normalize_list(x[0]), normalize_list(x[1])]
            l, a = model_p.train_on_batch(x, y)
            lp += l
            ap += a
            x, y = dataset_train_v[i]
            if not WRN: x = normalize_list(x)
            l, a = model_v.train_on_batch(x, y)
            lv += l
            av += a
            print('\rbatch {}/{} lp: {:4.2f} ap: {:4.2f} lv: {:4.2f} av: {:4.2f}'.format(t, n, lp/t, ap/t, lv/t, av/t), end='')
        print('')
        lossPval, accPval = model_p.evaluate_generator(dataset_valid_p)
        lossVval, accVval = model_v.evaluate_generator(dataset_valid_v)
        print('val, lp: {:4.2f} ap:{:4.2f}'.format(lossPval, accPval))
        print('val, lv: {:4.2f} av:{:4.2f}'.format(lossVval, accVval))
        if accPval > bestP:
            bestP = accPval
            if WRN: weights_w = {"network": WRN_28_4.state_dict()}
            else: weights_w = {"network": feature_extractor.state_dict()}
            torch.save(weights_w, filepath_w.format(epoch=e, val_acc=accPval))
            model_p.save_weights(filepath_p.format(epoch=e, val_acc=accPval))
        if accVval > bestV:
            bestV = accVval
            model_v.save_weights(filepath_v.format(epoch=e, val_acc=accVval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("-d", "--dataset", nargs=1)
    parser.add_argument("-e", "--nb_epochs", nargs=1)
    parser.add_argument("-f", "--size_frag", nargs=1)
    parser.add_argument("-i", "--path_dataset", nargs=1)
    parser.add_argument("-l", "--nb_help", nargs=1)
    parser.add_argument("-n", "--nb_frag", nargs=1)
    parser.add_argument("-p", "--puzzle_size", nargs=1)
    parser.add_argument("-s", "--space_size", nargs=1)
    parser.add_argument("-v", "--verbose", nargs=1)
    parser.add_argument("-w", "--wrn", nargs=1)
    parser.add_argument("-c", "--central_fragment", nargs=1)
    args = parser.parse_args()

    global VERBOSE, NB_EPOCHS, WRN
    global NB_FRAG, NB_HELP, CENTRAL_FRAGMENT
    global FRAG_SIZE, SPACE_SIZE, PUZZLE_SIZE
    global DATASET, STRUCT, DATASET_PATH

    VERBOSE = args.verbose[0] if args.verbose else False
    NB_EPOCHS = int(args.nb_epochs[0]) if args.nb_epochs else 50
    WRN = bool(int(args.wrn[0])) if args.wrn else True

    NB_FRAG = int(args.nb_frag[0]) if args.nb_frag else 3
    NB_HELP = int(args.nb_help[0]) if args.nb_help else 0
    CENTRAL_FRAGMENT = bool(args.central_fragment[0]) if args.central_fragment else False
    FRAG_SIZE = int(args.size_frag[0]) if args.size_frag else 40
    SPACE_SIZE = int(args.space_size[0]) if args.space_size else 4
    PUZZLE_SIZE = int(args.puzzle_size[0]) if args.puzzle_size else 3*40+2*4
    DATASET = "MET"

    DATASET_PATH = args.path_dataset[0] if args.path_dataset else "../datasets/MET/"
    STRUCT = str(FRAG_SIZE)+("-"+str(SPACE_SIZE)+"-"+str(FRAG_SIZE))*(NB_FRAG-1)+"_h"+str(NB_HELP)

    print('misc:', VERBOSE, NB_EPOCHS)
    print('archi is WRN:', WRN)
    print('fragments per side, helpers:', NB_FRAG, NB_HELP)
    print('structure:', STRUCT, DATASET)

    main()
