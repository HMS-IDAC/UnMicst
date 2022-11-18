import numpy as np
from os.path import exists, isfile, join, split, splitext, isdir
from os import makedirs, listdir
import tifffile
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import transform as trfm
from skimage.morphology import binary_erosion, disk
import shutil
import pandas as pd
import pickle
from torchvision.transforms import functional as F
import random
import torch.distributed as dist
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def createFolderIfNonExistent(path):
    if not exists(path): # from os.path
        makedirs(path)
        
def removeFolderIfExistent(path):
    if exists(path):
        shutil.rmtree(path)

def tifread(path):
    return tifffile.imread(path)

def tifwrite(I,path):
    tifffile.imsave(path, I)

def imshow(I,**kwargs):
    if not kwargs:
        plt.imshow(I,cmap='gray')
    else:
        plt.imshow(I,**kwargs)
        
    plt.axis('off')
    plt.show()

def imread(path):
    return skio.imread(path)

def imwrite(I,path):
    return skio.imsave(path,I)

def imerode(I,r):
    return binary_erosion(I, disk(r))

def im2double(I):
    if I.dtype == 'uint16':
        return I.astype('float64')/65535
    elif I.dtype == 'uint8':
        return I.astype('float64')/255
    elif I.dtype == 'float32':
        return I.astype('float64')
    elif I.dtype == 'float64':
        return I
    else:
        print('returned original image type: ', I.dtype)
        return I

def imresizeDouble(I,sizeOut): # input and output are double
    return trfm.resize(I,(sizeOut[0],sizeOut[1]),mode='reflect')
    
def listfiles(path,token): # path = folder path
    l = []
    for f in listdir(path):
        fullPath = join(path,f)
        if isfile(fullPath) and token in f:
            l.append(fullPath)
    l.sort()
    return l

def fileparts(path): # path = file path
    [p,f] = split(path)
    [n,e] = splitext(f)
    return [p,n,e]

def pathjoin(p,ne): # '/path/to/folder', 'name.extension' (or a subfolder)
    return join(p,ne)

def imadjust(I):
    I0 = I[I > 0]
    if np.any(I0):
        p1 = np.percentile(I0,1)
        p99 = np.percentile(I0,99)
    else:
        p1 = np.min(I)
        p99 = np.max(I)
        if p99 == 0:
            p99 = 1
    I = (I-p1)/(p99-p1)
    I[I < 0] = 0
    I[I > 1] = 1
    return I

def uint16Gray_to_uint8RGB(I):
    assert I.dtype == 'uint8'
    I = imadjust(I.astype('float64')/255)
    return np.uint8(255*np.stack([I,I,I],axis=2))

def uint16Gray_to_doubleGray(I):
    assert I.dtype == 'uint16'
    return imadjust(I.astype('float64')/65535)

def doubleGray_to_uint8RGB(I):
    assert I.dtype == 'float64'
    return np.uint8(255*np.stack([I,I,I],axis=2))

def saveData(data,path,verbose=False):
    if verbose:
        print('saving data')
    dataFile = open(path, 'wb')
    pickle.dump(data, dataFile)

def loadData(path,verbose=False):
    if verbose:
        print('loading data')
    dataFile = open(path, 'rb')
    return pickle.load(dataFile)

def writeTable(path,colTitles,matrix):
    # with open(path, 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(colTitles)
    #     for i in range(matrix.shape[0]):
    #         writer.writerow(matrix[i,:])

    T = {}
    for i in range(len(colTitles)):
        T[colTitles[i]] = matrix[:,i]
    df = pd.DataFrame(T)
    df.to_csv(path, index=False)

def readTable(path):
    # T = []
    # with open(path) as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         T.append(row)
    # colTitles = T[0]
    # nRows = len(T)-1
    # nCols = len(colTitles)
    # matrix = np.zeros((nRows,nCols))
    # for i in range(nRows):
    #     for j in range(nCols):
    #         matrix[i,j] = T[i+1][j]
    # return colTitles, matrix

    df = pd.read_csv(path)
    colTitles = df.columns.to_list()
    matrix = df.to_numpy()
    return colTitles, matrix

def imgaussfilt(I,sigma,**kwargs):
    return gaussian_filter(I,sigma,**kwargs)

def boxes_intersect(box_a, box_b):
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b

    min_ymax = np.minimum(ymax_a, ymax_b)
    max_ymin = np.maximum(ymin_a, ymin_b)
    min_xmax = np.minimum(xmax_a, xmax_b)
    max_xmin = np.maximum(xmin_a, xmin_b)

    x_intersection = np.maximum(min_xmax-max_xmin, 0)
    y_intersection = np.maximum(min_ymax-max_ymin, 0)
    
    return x_intersection*y_intersection > 0

def imfillholes(I):
    return binary_fill_holes(I)