import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):

    def __init__(self, encoder, projection_dim, dim_mlp=2048):
        super(ImageEncoder, self).__init__()

        self.encoder = encoder
        self.encoder.heads = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp, bias=True),
            nn.ReLU(),
            nn.Linear(dim_mlp, projection_dim, bias=True),
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return h1, h2, z1, z2

