from typing import Tuple, List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import my_gumbel_sinkhorn


class PuzzleNet(LightningModule):
    def __init__(self,
                 img_size=512,
                 block_size=64,
                 model_name='mobilenetv3_small_050',
                 learning_rate=0.01,
                 weight_decay=0.0001,
                 sinkhorn_iter=100,
                 sinkhorn_temp=0.5,
                 feature_size=256,
                 transformers_layers=2,
                 group_conv=256,
                 train_block_size=(512, 256, 128)
                 ):
        """
        :param img_size: For now, the image size is fixed and has to be passed to the network.
        :param block_size: This is the default block size, overwritten in the training step by one of the element of
        train_block_size.
        :param model_name: Name of the CNN encoder, must exist in the timm.list_models repository
        :param learning_rate:
        :param weight_decay:
        :param sinkhorn_iter: Number of iterations done in the Sinkhorn normalization
        :param sinkhorn_temp: Temperate $\tau$ applied in the Sinkhorn
        :param feature_size: Size of the feature vector at for each puzzle piece (aka: the size of the projection at the
        end of the encoder)
        :param transformers_layers: Number of Transformer Encoder Layer successive at the end of the encoder
        :param group_conv: Number of head in the cross-attention computation done to get the permutation matrix
        :param train_block_size: list of block sizes, one of which will be randomly chosen for each training step. This
        allows training with varying puzzle pieces.
        """
        super().__init__()

        if isinstance(img_size, list) or isinstance(img_size, tuple):
            assert img_size[0] == img_size[1], 'Only square images are accepted for now'
            img_size = img_size[0]

        assert img_size % block_size == 0, 'Image size is not divisible by the block size, ' \
                                           'this configuraiton is not implemented test'
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.img_size = img_size
        self.block_size = block_size
        self.feature_size = feature_size
        self.group_conv = group_conv
        self.n_transformers_layers = transformers_layers
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_temp = sinkhorn_temp
        self.train_block_sizes = train_block_size

        self.encoder = timm.create_model(model_name, pretrained=True, features_only=True)
        f_infos = self.encoder.feature_info.channels()
        f_size = f_infos[-1]
        self.proj_conv = nn.Sequential(nn.Conv2d(f_size, self.feature_size, 1), nn.ReLU())

        self.keys_queries_proj = nn.ModuleList(
            [nn.Conv2d(self.feature_size, self.feature_size, 1, groups=group_conv),
             nn.Conv2d(self.feature_size, self.feature_size, 1, groups=group_conv)])

        layers = [nn.Identity()]
        for i in range(self.n_transformers_layers):
            layers.append(nn.TransformerEncoderLayer(self.feature_size, 8, dim_feedforward=self.feature_size,
                                                     activation='gelu',
                                                     batch_first=True))
        self.fusion_module = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(self.group_conv, 1, 1)

        self.loss = nn.BCELoss()

        self.save_hyperparameters()

    @property
    def L(self) -> int:
        """
        :return: Number of puzzle pieces along a row/column

        """
        return self.img_size // self.block_size

    @property
    def n_pieces(self) -> int:
        """
        :return: Number of puzzle pieces
        """
        return self.L ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: torch.Tensor, size BXCxHxW
        :return: Permutation: torch.Tensor
        """

        # CNN part
        b = x.shape[0]
        y = self.encoder(x)[-1]  # Get last feature map
        y = F.interpolate(y, size=(self.L, self.L))
        y = self.proj_conv(y)

        # Transformer part
        y = y.view(b, self.feature_size, self.n_pieces)
        y = y.permute(0, 2, 1)
        A = self.fusion_module(y)
        A = A.permute((0, 2, 1)).view(b, self.feature_size, self.L, self.L)

        # Cross Attention part
        queries = self.keys_queries_proj[0](A)
        queries = queries.view(b, self.group_conv, self.feature_size // self.group_conv, self.n_pieces)
        queries = queries.permute(0, 1, 3, 2)

        keys = self.keys_queries_proj[1](A)
        keys = keys.view(b, self.group_conv, self.feature_size // self.group_conv, self.n_pieces)
        scale = self.feature_size ** -0.5

        cross_att = torch.matmul(queries, keys) * scale

        # Projection to permutation
        A = self.final_conv(cross_att)
        A, _ = my_gumbel_sinkhorn(A, noise_factor=0.0, temp=self.sinkhorn_temp, n_iters=self.sinkhorn_iter)
        return A

    def training_step(self, data, batch_index):
        """

        :param data:
        :param batch_index:
        :return: loss function
        """
        img = data['image']
        # Random sampling of the number of pieces
        self.block_size = np.random.choice(self.train_block_sizes)

        # Shuffling of the image
        unblock, indices = self.shuffle_image(img)

        # Estimation of the permutation
        y = self.forward(unblock).view(-1, self.n_pieces ** 2)

        # Shuffles indices to GT permutation matrix
        A_gt = self.permutation2mat(indices, device=img.device)

        # BCE Loss
        loss = self.loss(y, A_gt)
        self.log('Train loss', torch.nan_to_num(loss), on_step=True, on_epoch=True, sync_dist=True)

        return torch.nan_to_num(loss)

    def get_eye_gt(self, b):
        """
        Return the identity matrix, which corresponds to the permutation matrix on unshuffled image
        :param b: batch-size
        :return:
        """
        A_gt = torch.eye(self.n_pieces, device=self.device).unsqueeze(0)
        A_gt = A_gt.expand(b, -1, -1).view(b, -1)
        return A_gt

    def validation_step(self, data, batch_index):
        """
        Validation is done on unshuffle images
        :param data:
        :param batch_index:
        :return:
        """
        x = data['image']
        b = x.shape[0]
        y = self.forward(x).view(-1, self.L ** 4)
        A_gt = self.get_eye_gt(b)
        val_loss = self.loss(y, A_gt)
        self.log('Validation loss', val_loss, on_step=False, on_epoch=True, sync_dist=True)

    def permutation2mat(self, permutations: List[torch.Tensor], device=None) -> torch.Tensor:
        """
        Receive permutation as a list of tensor of indices.
        Return a torch of size BxLxL corresponding to the permutations matrix associated with these indices.
        :param permutations:
        :param device:
        :return:
        """
        if not isinstance(permutations, list):
            permutations = [permutations]
        A = []
        for perm in permutations:
            mat = torch.zeros((self.n_pieces, self.n_pieces), device=device)
            mat[torch.arange(self.n_pieces), perm] = 1
            A.append(mat.unsqueeze(0))
        A = torch.cat(A, 0)
        return A.view(-1, self.L ** 4)

    def shuffle_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Shuffling the image: we first unfold the batch of image into patches. Then for each image of the batch,
        we shuffle the patch dimension.
        Then we fold agaim the image.
        :param img:
        :return: shuffle images, indices to unshuffle the image
        """
        b = img.shape[0]
        block = F.unfold(img, (self.block_size, self.block_size), padding=0, stride=self.block_size)
        L = block.shape[-1]
        indices = []
        for i in range(b):
            index = torch.randperm(L)
            block[i] = block[i, :, index]
            indices.append(torch.argsort(index))

        unblock = F.fold(block, (self.img_size, self.img_size),
                         (self.block_size, self.block_size),
                         padding=0, stride=self.block_size)

        return unblock, indices

    def unshuffle_image(self, shuffled_image: torch.Tensor, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Reverse of the shuffle_image function
        :param shuffled_image:
        :param indices:
        :return:
        """
        block = F.unfold(shuffled_image, (self.block_size, self.block_size), padding=0, stride=self.block_size)
        b = block.shape[0]
        for i in range(b):
            block[i] = block[i, :, indices[i]]
        unblock = F.fold(block, (self.img_size, self.img_size),
                         (self.block_size, self.block_size),
                         padding=0, stride=self.block_size)
        return unblock

    def parameters(self, recurse: bool = True):
        return list(self.encoder.parameters()) + list(self.keys_queries_proj.parameters()) + list(
            self.fusion_module.parameters())

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-7)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def tenshow(tensor: torch.Tensor):
    import matplotlib.pyplot as plt
    tensor = tensor.squeeze().byte()
    img_npy = tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_npy)
    plt.show()


if __name__ == '__main__':
    m = PuzzleNet(model_name='seresnet50')
    foo = torch.rand(2, 3, 512, 512)
    output = m.forward(foo)
