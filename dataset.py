from enum import Enum
from typing import Tuple
import os
import os.path
import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'
    
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        self.is_train = mode == Dataset.Mode.TRAIN
        print("is_train= ", self.is_train)
        self._mnist = datasets.MNIST(path_to_data_dir, train=self.is_train, download=True)
        
        if self.is_train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(path_to_data_dir, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(path_to_data_dir, self.processed_folder, self.test_file))
        
    def __len__(self) -> int:
        # TODO: CODE START
        #raise NotImplementedError
        #print("len of self.mnist = ", len((self._mnist)))
        #print("len of self.mnist type= ", type(len(self._mnist)))
        return len(self._mnist)
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # TODO: CODE START
        #raise NotImplementedError
        #img, target = self._mnist[index]
        
        #target = self._mnist[index][1].item()
        transform1 = transforms.Compose([
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )
        
        if self.is_train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]
        img = transform1(self._mnist[index][0])
        #img = transform(img)
        #print(img.size())
        
        #print("img=", img)
        #print("img type= ", type(img))
        #print("target=", target)
        #print("target type= ", type(target))
        
        #raise NotImplementedError
        return img, label
        # TODO: CODE END

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
        return image
