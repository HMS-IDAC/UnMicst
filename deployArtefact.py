# -------------------------

# adapted from
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb
# by
# Marcelo Cicconet & Clarence Yapp

import os
import argparse
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
from skimage import morphology
from skimage.measure import label
import skimage.io
import tifffile
from ome_types import from_tiff
import glob
from scipy import arange
from toolbox.PartitionOfImageOM import PI2D
from skimage.transform import resize
from toolbox.toolbox import listfiles, tifread, uint16Gray_to_uint8RGB, imread, Compose, RandomHorizontalFlip, ToTensor, \
    get_transform, collate_fn, reduce_dict, imshow, fileparts, imwrite, imerode, imgaussfilt, \
    uint16Gray_to_doubleGray, doubleGray_to_uint8RGB,imfillholes

class CellsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, load_annotations=True,channel=0,scaling=1):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = listfiles(root, '.tif')
        self.ants = None
        self.channel = channel
        self.scaling = scaling
        if load_annotations:
            self.ants = listfiles(root, '.png')

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        raw = tifffile.imread(img_path, key=self.channel)
        img = uint16Gray_to_uint8RGB(raw)

        dsFactor = self.scaling
        hsize = int((float(img.shape[0]) * float(dsFactor)))
        vsize = int((float(img.shape[1]) * float(dsFactor)))
        img = np.uint8(resize(img.astype(float), (vsize, hsize), mode='reflect', order=0))
        target = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, raw

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes,path):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,box_detections_per_img=200)
    # model = torch.load(path)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="path to the .tif files")
    parser.add_argument("--segPath", help="path to the segmentation mask .tif files")
    parser.add_argument("--model",  help="type of model. For example, nuclei vs cytoplasm", default = 'artefact')
    parser.add_argument("--outputPath", help="output path of probability map")
    parser.add_argument("--channel", help="channel to perform inference on",  nargs = '+', default=[0])
    parser.add_argument("--threshold", help="threshold for filtering objects. Max is 1.", type = float, default=0.6)
    parser.add_argument("--overlap", help="amount of overlap when stitching. Default is 128.", type=int, default=128)
    parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
                        default=0.0625)
    parser.add_argument("--stackOutput", help="save probability maps as separate files", action='store_true')
    parser.add_argument("--GPU", help="explicitly select GPU", action='store_true')
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.realpath(__file__))
    modelPath = os.path.join(scriptPath, 'models', args.model, args.model + '.pt')
    deploy_path_in = args.imagePath
    deploy_path_out = args.outputPath
    channel = args.channel[0]

    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    ome = from_tiff(args.imagePath)
    numChan = ome.images[0].pixels.size_c

    if args.GPU:
        device_train = torch.device('cuda')
        print('using GPU')
    else:
        device_train = torch.device('cpu')
        print('using CPU')


    coco_path = os.path.join(scriptPath, 'models', args.model, 'cocomodel.pt')
    def get_boxes_and_contours(im, mk, bb, sc):
        boxes = []
        contours = []
        for i in range(bb.shape[0]):
            if sc[i] > args.threshold:
                x0, y0, x1, y1 = np.round(bb[i, :]).astype(int)
                x0 = int(x0)
                y0 = int(y0)
                x1 = int(x1)
                y1 = int(y1)
                x1 = np.minimum(x1, im.shape[1] - 1)
                y1 = np.minimum(y1, im.shape[0] - 1)

                if (y1 - y0) * (x1 - x0) < (im.shape[0] * im.shape[1] * 0.1):
                    boxes.append([x0, y0, x1, y1])

                    # maskSlice = resize(p[i,:,:], (sizeOut[0], sizeOut[1]), mode='reflect')
                    mask_box = np.zeros(im.shape, dtype=bool)
                    mask_box[y0:y1, x0:x1] = True

                    mask_i = np.logical_and(mk[i, :, :] > 0.6, mask_box)
                    mask_i = morphology.remove_small_holes(morphology.remove_small_objects(mask_i,30), 1000)
                    ct =np.logical_and(mask_i, np.logical_not(imerode(mask_i, 1)))
                    ct_coords = np.argwhere(ct)
                    contours.append(ct_coords)

        return boxes, contours

    num_classes = 2
    suggestedPatchSize =1024
    margin = args.overlap
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes,coco_path)
    # move model to the right device
    model.to(device_train)

    if args.GPU:
        model.load_state_dict(torch.load(modelPath))

    else:
        model.load_state_dict(torch.load(modelPath, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        model.to(device_train)
        file_path = args.imagePath
        _, file_name, _ = fileparts(file_path)
        print('processing image', file_name)
        fileName = os.path.basename(file_path)
        file_name = fileName.split(os.extsep, 1)

        dsFactor = args.scalingFactor
        hsize = int(float(ome.images[0].pixels.size_x * float(dsFactor)))
        vsize = int(float(ome.images[0].pixels.size_y * float(dsFactor)))
        a = np.empty((vsize, hsize, numChan), dtype=np.uint8)
        preview = np.empty((numChan,vsize, hsize), dtype=np.uint8)

        for iChan in range(numChan):
            img_tif = skimage.io.imread(args.imagePath, img_num=iChan, plugin='tifffile')
            img_double = uint16Gray_to_doubleGray(img_tif)
            img_double = (resize(img_double, (vsize, hsize), mode='reflect', order=0))
            PI2D.setup(img_double, suggestedPatchSize, margin)
            for i_patch in range(PI2D.NumPatches):
                P = PI2D.getPatch(i_patch)
                P3 = doubleGray_to_uint8RGB(P)
                img = torch.tensor(np.transpose(P3, [2, 0, 1]).astype(np.float32) / 255)
                prediction = model([img.to(device_train)])
                im = np.mean(img.numpy(), axis=0)
                mk = prediction[0]['masks'][:, 0].cpu().numpy()
                bb = prediction[0]['boxes'].cpu().numpy()
                sc = prediction[0]['scores'].cpu().numpy()

                boxes, contours = get_boxes_and_contours(im, mk, bb, sc)
                PI2D.patchOutput(i_patch, boxes, contours)
            PI2D.prepareOutput()
            labelMask = np.uint8(imfillholes(PI2D.Outputlabel))
            a[:,:,iChan] = labelMask
            test=PI2D.OutputBoxes

            img_double[PI2D.OutputBoxes>0] =1
            preview[iChan, :, :] = img_double*255
            print('Found ' + str(len(boxes)) + " objects in channel " + str(iChan+1))
        labelMask = np.amax(a,axis=2)
        labelMaskFR = (resize(labelMask,(ome.images[0].pixels.size_y,ome.images[0].pixels.size_x))>0)
        skimage.io.imsave(
            args.outputPath + '//' + file_name[0] + '_Preview.tif'
            , np.uint32(preview))

        # load segmentation mask
        # segMask = tifread(args.segPath)
        # hsize = segMask.shape[2]
        # vsize = segMask.shape[1]
        # labelMaskFR = resize(labelMask,(vsize,hsize))
        # segMask[2,:,:] = segMask[2,:,:] + (255*labelMaskFR)
        # for iClass in range(2):
        #     slice = segMask[iClass, :, :]
        #     slice[labelMaskFR>0] =0
        #     segMask[iClass,:, :] = slice

        save_kwargs = {
            'bigtiff': True,
            'metadata': None,
            'append': True,
        }
        skimage.io.imsave(args.segPath, np.uint8(255*labelMaskFR), **save_kwargs)







