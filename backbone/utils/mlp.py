
import torch.nn as nn
from backbone.utils.k_winners import KWinners1d, KWinners2d

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kw_percent_on=1, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = KWinners1d(
            channels=hidden_features,
            percent_on=kw_percent_on,
            relu=True
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        print("Inside MLP")
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.act(x)
        print("=" *30)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
