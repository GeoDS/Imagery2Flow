import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):

    def __init__(self, encoder, projection_dim, dim_mlp=2048):
        super(ImageEncoder, self).__init__()

        self.encoder = encoder
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=True), nn.ReLU(), self.encoder.fc)

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return h1, h2

