
from torch.utils.data.dataloader import DataLoader
import argparse
import albumentations as A #For applying same transforms
import os #for creating and removing directories
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import csv
import matplotlib.pyplot as plt
import random
import os
import matplotlib.pyplot as plt
import glob

#RAFDB Image Class
class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)

#Replace image file paths if using RAFDB
class RafDataset(data.Dataset):
  def __init__(self, unmask = '/content/drive/MyDrive/FERDatasets/RAFDB/aligned', mask_file = '/content/drive/MyDrive/Mask_RAFDB/aligned_mask', partition = 'train', transform = None, num_classes = 7, loader = my_loader):
    """
      root --> path to ocation of images in drive
      mask_file --> path for masked RAF-DB images
      num_classes --> number of annotations (7 in RAF-DB)
      labels are same for both masked and unmasked
      Need mask label as well for mask detection task

      Note: We read the labels here but leave reading of images to __getitem__
    """
    self.unmask = unmask
    self.mask_file = mask_file
    self.bin = '/content/drive/MyDrive/Face_Inpainting_1/Binary_RAFDB'
    self.num_classes = num_classes
    self.partition = partition
    self.loader = loader
    self.transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    if partition == 'train':


      self.transform1 = A.Compose([
                                  A.HueSaturationValue(), A.RandomContrast() #Removed Horizontal flip
                                 #A.ShiftScaleRotate(shift_limit = 0.0625,scale_limit = 0.1 ,rotate_limit = 3, p = 0.5),
                                 #A.IAAAffine(scale = (1.0, 1.25), rotate = 0.0, p = 0.5)
                                 ],
                                  additional_targets={'image1':'image'})
      self.transform2 = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                            ])



    NAME_COLUMN = 0
    LABEL_COLUMN = 1
    #Reading labels
    train_txtfile = pd.read_csv('/content/drive/MyDrive/FERDatasets/RAFDB/train_label.txt', sep=' ', header=None)
    test_txtfile = pd.read_csv('/content/drive/MyDrive/FERDatasets/RAFDB/test_label.txt', sep=' ', header=None)
    if partition == 'train':
      self.label = train_txtfile.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
      file_names = train_txtfile.iloc[:, NAME_COLUMN].values
    else:
      self.label = test_txtfile.iloc[:, LABEL_COLUMN].values - 1 #same as above
      file_names = test_txtfile.iloc[:, NAME_COLUMN].values

    #Changing file names to match actual image names

    self.unmasked_file_paths = []
    self.masked_file_paths = []
    self.dataset = []
    #print(set(self.label))
    i = 0
    """
    L = []
    img1 = '/content/drive/MyDrive/FERDatasets/RAFDB/aligned/train_00057_aligned.jpg'
    img2 = '/content/drive/MyDrive/Mask_RAFDB/aligned_mask/train_00057_aligned.jpg'
    img3 = '/content/drive/MyDrive/Face_Inpainting_1/Binary_RAFDB/train_00057_aligned.jpg'
    for j in range(0, 7):
      L.append(j)
      item = (img1, img2, img3)
      self.dataset.append(item)
    #print(L)
    """
    L = []



    for f in file_names:
        #if self.label[i] != 6:#Surprise, F, D, H, sad, a, n
          #i+=1
          #continue
        if "test_0215" in f: #Faulty binary map
          i += 1
          continue
        f = f.split(".")[0]
        f = f +"_aligned.jpg"
        #working_directory = self.unmask + aligned1
        #Putting non-masked image paths into unmasked_file_paths
        upath = os.path.join(self.unmask, f)
        self.unmasked_file_paths.append(upath)
        #Putting masked image paths into masked_file_paths
        #working_directory = file2 + aligned2
        mpath = os.path.join(self.mask_file, f)
        self.masked_file_paths.append(mpath)
        #Binary mask
        bpath = os.path.join(self.bin, f)
        item = (upath, mpath, bpath)
        self.dataset.append(item)
        L.append(self.label[i])
        i += 1

    #print(j)
    self.label = L

  def __getitem__(self, index):
    """
    Here we read the actual images
    We randomly apply few transforms to the image for image augmentation
    Return: image1, image 2 and their label
    """

    uimg_path, mimg_path, b = self.dataset[index]
    #print(mimg_path)
    uimg = cv2.imread(uimg_path)
    uimg = cv2.cvtColor(uimg, cv2.COLOR_BGR2RGB)
    mimg = cv2.imread(mimg_path)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_BGR2RGB)
    bimg = my_loader(b)
    bimg = self.transform(bimg)
    #print(bimg.shape)
    #bimg = bimg.permute(1,2,0)

    """
    Note on cv2.imread(): OpenCV uses the BGR format
    We need to change to RGB format(ideally before returning the image)
    """
    if self.partition == 'train':
      #transformed = self.transform1(image = uimg, image1 = mimg)
      #uimg  = transformed['image']
      #mimg = transformed['image1']
      uimg = self.transform2(uimg)
      mimg = self.transform2(mimg)
    else:
      uimg = self.transform(uimg)
      mimg = self.transform(mimg)
    label = self.label[index]
    name = b.split('/')[-1]
    return uimg, mimg, bimg, label, name

  def __len__(self):
    return len(self.dataset)

  def show_img(self, index):
    uimg, mimg, bimg, label, name = self.__getitem__(index)
    uimg = uimg.permute(1, 2, 0)
    mimg = mimg.permute(1, 2, 0)
    bimg = bimg.permute(1,2,0)

    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(np.rot90(uimg,0))
    f.add_subplot(1,3, 2)
    plt.imshow(np.rot90(mimg,0))
    f.add_subplot(1,3,3)
    plt.imshow(np.rot90(bimg, 0))
    plt.show(block=True)

    print("Label: ", label)
    print(name)

#End of Class


