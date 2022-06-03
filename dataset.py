import os
import random
import torch
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import skimage.io as io

# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(64, 64)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)

    return image, label


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    y = Image.open(filepath).convert('L')
    return y


class DatasetFromFolder(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, thick_gt_dir, thin_gt_dir, ex=1):
        super(DatasetFromFolder, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)]) * ex
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)]) * ex
        self.thick_image_filenames = sorted(
            [os.path.join(thick_gt_dir, z) for z in os.listdir(thick_gt_dir) if is_image_file(z)]) * ex
        self.thin_image_filenames = sorted(
            [os.path.join(thin_gt_dir, w) for w in os.listdir(thin_gt_dir) if is_image_file(w)]) * ex

        self.LR_transform = x_transforms
        self.HR_transform = y_transforms
        self.thick_transform = x_transforms
        self.thin_transform = y_transforms

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])
        labels_thick = load_img(self.thick_image_filenames[index])
        labels_thin = load_img(self.thin_image_filenames[index])

        # avg_gray = np.average(labels)
        # im_gray2 = np.where(labels[..., :] < avg_gray, 0, 1)
        # labels = np.array(im_gray2, dtype=np.int8)
        gt = np.array(labels)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        labels = Image.fromarray(gt)
        # labels = np.array(labels, dtype=np.uint8)
        # labels[labels < 128] = 0
        # labels[labels >= 128] = 1
        # labels = Image.fromarray(np.array(labels))
        rotate = 10
        angel = random.randint(-rotate, rotate)
        inputs = inputs.rotate(angel)
        labels = labels.rotate(angel)
        labels_thick = labels_thick.rotate(angel)
        labels_thin = labels_thin.rotate(angel)

        labels = self.HR_transform(labels)
        inputs = self.LR_transform(inputs)
        labels_thick = self.thick_transform(labels_thick)
        labels_thin = self.thin_transform(labels_thin)

        return inputs, labels, labels_thin, labels_thick

    def __len__(self):
        return len(self.LR_image_filenames)


x_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize(256),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize(256)
    # transforms.Normalize([0.5], [0.5])
])


class DatasetFromFolder1(Dataset):
    def __init__(self, LR_image_dir):
        super(DatasetFromFolder1, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)])
        self.LR_transform = x_transforms

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        LR_images = self.LR_transform(inputs)
        return LR_images

    def __len__(self):
        return len(self.LR_image_filenames)


class DatasetFromFolder2(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, ex=1):
        super(DatasetFromFolder2, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)]) * ex
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)]) * ex

        self.LR_transform = x_transforms
        self.HR_transform = y_transforms

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])

        gt = np.array(labels)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        labels = Image.fromarray(gt)

        rotate = 10
        angel = random.randint(-rotate, rotate)
        inputs = inputs.rotate(angel)
        labels = labels.rotate(angel)

        #inputs, labels = random_crop(inputs, labels)

        inputs = self.LR_transform(inputs)
        labels = self.HR_transform(labels)
        return inputs, labels

    def __len__(self):
        return len(self.LR_image_filenames)


# test_dataset = DatasetFromFolder1("./datasets/ROSE-2/test/8bit_gt/")
# dataloaders = DataLoader(test_dataset, batch_size=1)

class DatasetFromFolderOCTA(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, pred_dir, ex=1):
        super(DatasetFromFolderOCTA, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)]) * ex
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)]) * ex
        self.pred_image_filenames = sorted(
            [os.path.join(pred_dir, z) for z in os.listdir(pred_dir) if is_image_file(z)]) * ex



        self.LR_transform = x_transforms
        self.HR_transform = y_transforms

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])
        inputs1 = load_img(self.pred_image_filenames[index])


        gt = np.array(labels)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        labels = Image.fromarray(gt)

        rotate = 10
        angel = random.randint(-rotate, rotate)
        inputs = inputs.rotate(angel)
        labels = labels.rotate(angel)
        inputs1 = inputs1.rotate(angel)
        #inputs, labels = random_crop(inputs, labels)

        inputs = self.LR_transform(inputs)
        labels = self.HR_transform(labels)
        inputs1 = self.HR_transform(inputs1)
        return inputs, labels, inputs1

    def __len__(self):
        return len(self.LR_image_filenames)


class DatasetFromFolder_test(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, ex=1):
        super(DatasetFromFolder_test, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)]) * ex
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)]) * ex

        self.LR_transform = x_transforms
        self.HR_transform = y_transforms

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])

        gt = np.array(labels)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        labels = Image.fromarray(gt)

        #inputs, labels = random_crop(inputs, labels)

        inputs = self.LR_transform(inputs)
        labels = self.HR_transform(labels)
        return inputs, labels

    def __len__(self):
        return len(self.LR_image_filenames)