import argparse
import torch
from lib.puzzles_generator_MET import prepare_data_p

from lib.vit_pytorch import ViT

import pytorch_lightning as pl


def main():
    print("loading datasets")
    ##### Prepare data for P ########
    dataset_train_p = prepare_data_p(path=DATASET_PATH, phase="train", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT,
                                     batch_size=1, data_aug=AUGMENT)
    dataset_valid_p = prepare_data_p(path=DATASET_PATH, phase="val", puzzle_size=PUZZLE_SIZE, fragment_per_side=NB_FRAG,
                                     fragment_size=FRAG_SIZE, space=SPACE_SIZE, nb_helpers=NB_HELP, central_known=CENTRAL_FRAGMENT,
                                     batch_size=1)
    print("p dataset has been loaded")



    x,_ = dataset_train_p[0]
    img_size = x.shape[2]
    frg_size = FRAG_SIZE+SPACE_SIZE

    ### Prepare neural network P ####

    model = ViT(
        image_size=img_size,
        patch_size=frg_size,
        num_classes=NB_FRAG**2,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

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

    model_p = LitModelP()

    train = torch.utils.data.DataLoader(dataset_train_p, batch_size=128, num_workers=6)
    val = torch.utils.data.DataLoader(dataset_valid_p, batch_size=64, num_workers=6)

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='model_p_'+STRUCT)
    trainer = pl.Trainer(gpus=1, logger=logger, weights_summary='full', max_epochs=NB_EPOCHS)
    trainer.fit(model_p, train_dataloader=train, val_dataloaders=val)
    trainer.save_checkpoint(logger.log_dir + '/model_p_final_checkpoint.ckpt')


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
    args = parser.parse_args()

    global VERBOSE, NB_EPOCHS, WRN
    global NB_FRAG, NB_HELP, CENTRAL_FRAGMENT
    global FRAG_SIZE, SPACE_SIZE, PUZZLE_SIZE
    global DATASET, STRUCT, DATASET_PATH
    global AUGMENT
    global CONV_HEAD

    CONV_HEAD = args.conv_head
    AUGMENT = args.augment

    VERBOSE = args.verbose[0] if args.verbose else False
    NB_EPOCHS = int(args.nb_epochs[0]) if args.nb_epochs else 50

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
    print('conv:', CONV_HEAD)
    print('fragments per side, helpers:', NB_FRAG, NB_HELP)
    print('structure:', STRUCT, DATASET)

    main()
