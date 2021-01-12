from random import shuffle
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, mode = 'train'):
        self.imList = imList
        self.labelList = labelList
        self.mode = mode
        self.shuffle = shuffle

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):

        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = Image.open(image_name)
        label = Image.open(label_name)

        Transform = []
        Transform.append(T.Resize((256, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        label = Transform(label)

        image = image.float()
        label = label.float()

        Norm_ = T.Normalize(([0.5]), ([0.5]))
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        return image, label


def get_loader(imList, labelList, batch_size, num_workers=1, mode='train', drop_last=True):
    """Builds and returns Dataloader."""

    dataset = MyDataset(imList, labelList, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader