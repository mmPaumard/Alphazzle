import torch
import torch.nn as nn

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
