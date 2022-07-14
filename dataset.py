import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL
import torchvision
import torchvision.transforms.functional as tf
import numpy as np

import random
from typing import Union, cast
import config
from MaskImage import MaskedImage

from glob import glob
from tqdm import tqdm
import pathlib
import os
from torch.nn.functional import normalize


def extract_and_random_crop(real_imgs_path: str,
                            cropped_imgs_path: str = 'data/cropped_imgs'):
    images_paths = [pathlib.Path(img_path)
                    for img_path in glob(f'{real_imgs_path}/**/*.jpg')]
    for image_pth in tqdm(images_paths, total=len(images_paths), unit="images"):
        cropped_folder, cropped_filename = image_pth.parts[-2:]

        try:
            os.makedirs(os.path.join(cropped_imgs_path, cropped_folder))
        except FileExistsError:
            pass

        cropped_path = os.path.join(
            cropped_imgs_path, cropped_folder, cropped_filename)

        random_offset, random_spacing = tuple([random.randint(config.MIN_OFFSET, config.MAX_OFFSET) for _ in range(
            2)]), tuple([random.randint(config.MIN_SPACING, config.MAX_SPACING) for _ in range(2)])

        image_pil = Image.open(image_pth)
        image_array = np.asarray(image_pil)
        cropped_array, _, _ = MaskedImage(
            image_array, random_offset, random_spacing)
        image_cropped = Image.fromarray(cropped_array.transpose((2, 0, 1)))
        image_cropped.save(cropped_path)


class ImageRestorationDataset(Dataset):

    def __init__(self, img_dir='data'):
        super(ImageRestorationDataset, self).__init__()
        self.real_paths = [pathlib.Path(img_path) for img_path in glob(
            f'{img_dir}/real_imgs/**/*.jpg')]
        self.cropped_paths = [pathlib.Path(img_path) for img_path in glob(
            f'{img_dir}/cropped_imgs/**/*.jpg')]
        self.transformation = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((100, 100))])

    def transform(self, x):
        #x = torch.from_numpy(x)
        x = tf.to_tensor(x)
        x = tf.resize(x, config.IMAGE_SIZE,
                      interpolation=tf.InterpolationMode.BILINEAR)
        return x

    def normalize(self, image_data):
        mean = image_data.mean()
        std = image_data.std()
        image_data[:] -= mean
        image_data[:] /= std
        return image_data

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx: int):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_offset, random_spacing = tuple([random.randint(config.MIN_OFFSET, config.MAX_OFFSET) for _ in range(
            2)]), tuple([random.randint(config.MIN_SPACING, config.MAX_SPACING) for _ in range(2)])

        opened_real = Image.open(self.real_paths[idx])
        img_cropped, known_array, _ = MaskedImage(np.asarray(
            opened_real).astype(np.uint8), random_offset, random_spacing)

        img_cropped = np.concatenate(
            (img_cropped, np.expand_dims(known_array[0, :, :], 0)), 0)
        img_cropped = Image.fromarray(img_cropped.transpose((1, 2, 0)))

        resized_cropped = self.transform(img_cropped)
        resized_real = self.transform(opened_real)

        return resized_cropped, resized_real


# if __name__ == '__main__':
#     print(ImageRestorationDataset()[0][0].shape,
#           ImageRestorationDataset()[0][1].shape)
