import os
import time

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: CODE START
        #raise NotImplementedError
                # Define the parameters in your network.
        # This is achieved by defining the shapes of the multiple layers in the network.

        # Define two 2D convolutional layers (1 x 10, 10 x 20 each)
        # with convolution kernel of size (5 x 5).
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Define a dropout layer
        self.conv2_drop = nn.Dropout2d()

        # Define a fully-connected layer (320 x 10)
        self.fc = nn.Linear(320, 10)
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tensor:
        # TODO: CODE START
        images = F.relu(F.max_pool2d(self.conv1(images), 2))

        # ... -> Conv2 (64 x 20 x 8 x 8) -> Dropout -> Max Pooling (64 x 20 x 4 x 4) -> ReLU -> ...
        images = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(images)), 2))

        # ... -> Flatten (64 x 320) -> ...
        images = images.view(-1, 320)

        # ... -> FC (64 x 10) -> ...
        images = self.fc(images)

        # ... -> Log Softmax -> Output
        return F.log_softmax(images, dim=1)
        #raise NotImplementedError
        # TODO: CODE END

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        # TODO: CODE START
        #print("logist= ",logits)
        #print("labels= ", labels)
        #print("logist size= ",logits.size())
        #print("labels size= ", labels.size())
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        #print("loss in loss= ", loss)
        return loss
        #raise NotImplementedError
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
