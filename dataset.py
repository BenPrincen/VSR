import albumentations as A
import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random


class RealVSR(Dataset):
    def __init__(self, root_dir):
        super(RealVSR, self).__init__()
        self.lq_data = []
        self.root_dir = root_dir
        self.lq_path = os.path.join(root_dir, "LQ")
        self.class_names = os.listdir(self.lq_path)
        for index, name in enumerate(self.class_names):
            lq_files = os.listdir(os.path.join(self.lq_path, name))[:3]
            self.lq_data += list(zip(lq_files, [index] * len(lq_files)))

    def __len__(self):
        return len(self.lq_data)

    def __getitem__(self, index):
        img_file, label = self.lq_data[index]
        lq_dir = os.path.join(self.lq_path, self.class_names[label])
        gt_dir = ["GT" if elem == "LQ" else elem for elem in lq_dir.split(os.sep)]
        gt_dir = os.sep.join(gt_dir)

        lq_image = np.array(Image.open(os.path.join(lq_dir, img_file)))
        gt_image = np.array(Image.open(os.path.join(gt_dir, img_file)))

        img_tf = config.both_transforms(image=lq_image)
        replay_dict = img_tf["replay"]
        lq_image = img_tf["image"]
        gt_image = A.ReplayCompose.replay(replay_dict, image=gt_image)["image"]

        low_res = config.lowres_transform(image=lq_image)["image"]
        high_res = config.highres_transform(image=gt_image)["image"]

        return low_res, high_res


def test():
    dataset = RealVSR(root_dir="RealVSR/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    print(len(dataset))

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
