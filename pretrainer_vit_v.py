import argparse
import torch
from lib.puzzles_generator_MET import prepare_data_v

from lib.nn_architectures import LitModelV

import pytorch_lightning as pl


def main():
    print("loading datasets")
    ##### Prepare data for P ########
    dataset_train_p = prepare_data_v(path=DATASET_PATH, phase="train", fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT,
                                     batch_size=1, data_aug=AUGMENT)
    dataset_valid_p = prepare_data_v(path=DATASET_PATH, phase="val", fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT,
                                     batch_size=1)
    print("v dataset has been loaded")



    x,_, _ = dataset_train_p[0]
    img_size = x.shape[2]

    if LOAD_WEIGHT:
        model_v = LitModelV.load_from_checkpoint(LOAD_WEIGHT, img_size=img_size, patch_size=FRAG_SIZE, space_size=SPACE_SIZE, CONV_HEAD=CONV_HEAD)
        print('v weights loaded from {}'.format(LOAD_WEIGHT))
    else:
        model_v = LitModelV(img_size, FRAG_SIZE, SPACE_SIZE, CONV_HEAD)

    train = torch.utils.data.DataLoader(dataset_train_p, batch_size=256, num_workers=6)
    val = torch.utils.data.DataLoader(dataset_valid_p, batch_size=32, num_workers=6)

    import matplotlib.pyplot as plt


    logger = pl.loggers.TensorBoardLogger('tb_logs', name='model_v_'+STRUCT)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='model_v-{epoch:03d}-{val_loss:.3f}', save_top_k=3, mode='min')
    trainer = pl.Trainer(gpus=1, logger=logger, weights_summary='full', max_epochs=NB_EPOCHS, callbacks=[checkpoint_callback])
    trainer.fit(model_v, train_dataloader=train, val_dataloaders=val)
    trainer.save_checkpoint(logger.log_dir + '/model_v_final_checkpoint.ckpt')


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
    parser.add_argument("-c", "--central_fragment", nargs=1)
    parser.add_argument("-a", "--augment", action="store_true", default=False)
    parser.add_argument("-o", "--conv_head", action="store_true", default=False)
    parser.add_argument("-d", "--load", nargs=1)
    args = parser.parse_args()

    global VERBOSE, NB_EPOCHS, WRN
    global NB_FRAG, NB_HELP, CENTRAL_FRAGMENT
    global FRAG_SIZE, SPACE_SIZE, PUZZLE_SIZE
    global DATASET, STRUCT, DATASET_PATH
    global AUGMENT
    global CONV_HEAD
    global LOAD_WEIGHT

    CONV_HEAD = args.conv_head
    AUGMENT = args.augment
    VERBOSE = args.verbose[0] if args.verbose else False
    NB_EPOCHS = int(args.nb_epochs[0]) if args.nb_epochs else 500
    LOAD_WEIGHT = args.load[0] if args.load else False

    NB_FRAG = int(args.nb_frag[0]) if args.nb_frag else 3
    NB_HELP = int(args.nb_help[0]) if args.nb_help else 0
    CENTRAL_FRAGMENT = bool(args.central_fragment[0]) if args.central_fragment else False
    FRAG_SIZE = int(args.size_frag[0]) if args.size_frag else 40
    SPACE_SIZE = int(args.space_size[0]) if args.space_size else 4
    PUZZLE_SIZE = int(args.puzzle_size[0]) if args.puzzle_size else 3*40+2*4
    DATASET = "MET"

    DATASET_PATH = args.path_dataset[0] if args.path_dataset else "../datasets/MET/"
    STRUCT = str(FRAG_SIZE)+("-"+str(SPACE_SIZE)+"-"+str(FRAG_SIZE))*(NB_FRAG-1)+"_h"+str(NB_HELP)

    print('misc:', VERBOSE, NB_EPOCHS, AUGMENT)
    print('arch:', CONV_HEAD)
    print('fragments per side, helpers:', NB_FRAG, NB_HELP)
    print('structure:', STRUCT, DATASET)

    main()
