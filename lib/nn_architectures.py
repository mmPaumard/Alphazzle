import torch
import torch.nn as nn
import pytorch_lightning as pl
from lib.vit_pytorch import ViT
from torchvision.models import resnet18
from torchvision import transforms

# model = ViT(
#     image_size=img_size,
#     patch_size=frg_size,
#     num_classes=NB_FRAG ** 2,
#     dim=1024,
#     depth=6,
#     heads=8,
#     mlp_dim=2048
# )


# lightning wrapper
class LitModelV(pl.LightningModule):

    def __init__(self, img_size, patch_size, space_size, CONV_HEAD):
        super().__init__()
        self.vit = ViT(image_size=img_size,
                       patch_size=patch_size,
                       space_size=space_size,
                       num_classes=1,
                       dim=128,
                       depth=8,
                       heads=8,
                       mlp_dim=256,
                       conv_head=CONV_HEAD,
                       dropout=0.,
                       emb_dropout=0.0)
        # self.vit = resnet18(pretrained=True)
        # self.vit.train(False)
        # self.vit.requires_grad = False
        # self.vit.fc = nn.Identity()
        # # self.resnet.requires_grad = False
        # self.rdim = self.vit(torch.randn(1, 3, 64, 64)).shape[1]
        # print('rdim: {}'.format(self.rdim))
        # self.vit.fc = nn.Linear(self.rdim, 1)
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])

        self.example_input_array = torch.randn((1, 3, img_size, img_size))
        self.learning_rate = 3e-5

    def forward(self, x, mask=None):
        # in lightning, forward defines the prediction/inference actions
        # x = self.normalize(x)
        if mask is not None:
            # print(mask.view(x.shape[0],-1).sum(dim=1))
            if mask.sum().item() > 0:
                mask = mask.bool()
            else:
                mask = None
        out = self.vit(x, mask)
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, mask = batch
        y_hat = self.forward(x, mask)
        y = y.type_as(y_hat)
        # loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        l2 = torch.nn.functional.mse_loss(y_hat, y)
        l1 = torch.nn.functional.l1_loss(y_hat, y)
        loss = l2 + l1
        y_hat = (y_hat > 0.5).float()
        acc = (y_hat == y).sum().item()/x.shape[0]
        # Logging to TensorBoard by default
        self.log('train_l1', l1, on_epoch=True, on_step=False, logger=True)
        self.log('train_l2', l2, on_epoch=True, on_step=False, logger=True)
        self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x, mask)
        y = y.type_as(y_hat)
        # loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        l2 = torch.nn.functional.mse_loss(y_hat, y)
        l1 = torch.nn.functional.l1_loss(y_hat, y)
        loss = l2 + l1
        y_hat = (y_hat > 0.5).float()
        acc = (y_hat == y).sum().item()/x.shape[0]
        # Logging to TensorBoard by default
        self.log('val_l1', l1, on_epoch=True, on_step=False, logger=True)
        self.log('val_l2', l2, on_epoch=True, on_step=False, logger=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('val_acc', acc, on_epoch=True, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LitModelP(pl.LightningModule):

    def __init__(self, img_size, frg_size, space_size, CONV_HEAD, NB_FRAG):
        super().__init__()
        self.vit = ViT(image_size=img_size,
                       patch_size=frg_size,
                       space_size=space_size,
                       num_classes=NB_FRAG**2,
                       dim=128,
                       depth=8,
                       heads=8,
                       mlp_dim=256,
                       conv_head=CONV_HEAD,
                       dropout=0.,
                       emb_dropout=0.)

        self.example_input_array = torch.randn((1, 3, img_size, img_size))
        self.learning_rate = 3e-5

    def forward(self, x, mask=None):
        # in lightning, forward defines the prediction/inference actions
        if mask is not None:
            mask = mask.bool()
        out = self.vit(x, mask)
        if mask is not None:
            # print('mask1: {}'.format(mask))
            mask = mask[..., 1:, 1:].flatten(1)
            # print('mask: {}'.format(mask))
            out.masked_fill_(mask, float(-100.0))
            # print('out: {}'.format(out))
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, mask = batch
        bs = x.shape[0]
        y_hat = self.forward(x, mask)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        _, amax = torch.max(y_hat, 1)
        acc = (amax == y).float().sum().item()/x.shape[0]
        sor, arg = torch.sort(y_hat, descending=True)
        rk = torch.zeros(bs)
        for i in range(bs):
            rk[i] = arg[i, y[i]]
        map = (1./(1.+rk)).sum().item()/bs
        rank = rk.sum().item()/bs
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)
        self.log('train_map', map, on_epoch=True, on_step=False, logger=True)
        self.log('train_rank', rank, on_epoch=True, on_step=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, mask = batch
        bs = x.shape[0]
        y_hat = self.forward(x, mask)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        _, amax = torch.max(y_hat, 1)
        acc = (amax == y).float().sum().item()/x.shape[0]
        sor, arg = torch.sort(y_hat, descending=True)
        rk = torch.zeros(bs)
        for i in range(bs):
            rk[i] = arg[i, y[i]]
        map = (1./(1.+rk)).sum().item()/bs
        rank = rk.sum().item()/bs
        # Logging to TensorBoard by default
        self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('val_acc', acc, on_epoch=True, on_step=False, logger=True)
        self.log('val_map', map, on_epoch=True, on_step=False, logger=True)
        self.log('val_rank', rank, on_epoch=True, on_step=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ModelP(nn.Module):
    def __init__(self, wrn, nb_out, img_size=6400, frg_size=1024):
        super(ModelP, self).__init__()
        self.wrn = wrn
        self.feature_extractor1 = nn.Sequential(
            self.wrn,
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.feature_extractor2 = nn.Sequential(
            self.wrn,
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(frg_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.pred = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,nb_out),
            #nn.Softmax()
        )

    def forward(self, x1, x2):
        xf1 = self.feature_extractor1(x1)
        xf2 = self.feature_extractor2(x2)
        x = torch.cat([xf1, xf2], 1)
        return self.pred(x)



class ModelPPretrained(nn.Module):
    def __init__(self, base, nb_out, img_size=6400, frg_size=1024):
        super(ModelPPretrained, self).__init__()
        self.base = base
        self.feature_extractor1 = nn.Sequential(
            self.base,
            nn.Linear(img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.feature_extractor2 = nn.Sequential(
            self.base,
            nn.Linear(frg_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.pred = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,nb_out),
            #nn.Softmax()
        )

    def forward(self, x1, x2):
        xf1 = self.feature_extractor1(x1)
        xf2 = self.feature_extractor2(x2)
        x = torch.cat([xf1, xf2], 1)
        return self.pred(x)
