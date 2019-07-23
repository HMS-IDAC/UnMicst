# Image Preprocessing {#preprocess}

Images can be preprocessed by inferring nuclei contours via a pretrained UNet model. The model is trained on 3 classes : background, nuclei contours and nuclei centers. The resulting probability maps can then be loaded into any modular segmentation pipeline that may use (but not limited to) a marker controlled watershed algorithm. 

The only **input** file is:
an .ome.tif or .tif  (preferably flat field corrected, minimal saturated pixels, and in focus. The model is trained on images acquired at 20x with binning 2x2 or a pixel size of 0.65 microns/px. If your settings differ, you can upsample/downsample to some extent.

**How to install:**
1. Copy the python script, UNet model, and ImageScience toolbox to your computer. Clone from https://github.com/HMS-IDAC/UNet4Sage.git
2. Pip install tensorflow (or tensorflow_gpu with CUDA drivers and CuDNN libraries), matplotlib, scikit-image, Pillow, tifffile, Image, scipy

**How to run:**
3. Open the python script batchUNet2DtCycif.py in an editor.
4. Make the following changes to the code to reflect the locations of your data and supporting files:
-line 10 update the path to the ImageScience toolbox folder `sys.path.insert(0, 'path//to//UNet code//ImageScience')`
-line 509 update the path to the model `modelPath = 'modelPath = 'path//to//UNet code//TFModel - 3class 16 kernels 5ks 2 layers'`
-line 515 update the path to the top level experiment folder of the data `imagePath = 'path//to//parent//folder//of//data'`
your files should be stored in a subfolder called `registration` 
-line 516 : if you have multiple samples and they have a similar prefix, add the prefix/suffix here: `sampleList = glob.glob(imagePath + '//105*')`
-line 520 : if your files have a different extension from **tif**, you can change the extension here:
`fileList = glob.glob(iSample + '//registration//*.tif')`

5. **some helpful tips:**
-line 517 - specify the channel to infer nuclei contours and centers. If you want to run UNet on the 1st channel (sometimes DAPI/Hoechst), put 0.
-line 518 - if you acquired your images at a higher magnification (ie. 40x), you may want to downsample your image so that it is more similar to the trained model (ie. 20x binning 2x2, pixel size 0.65 microns).

6. in your terminal, activate your virtual environment and run this python script:
`python batchUNet2DtCycif.py`

7. If using tensorflow-gpu, your GPU card should be found. If not, prepare to hear your CPU fan fly! 
8. The probabilty map for the contours will be saved as a 3D tif file (concatenated with the original channel) and saved in a subfolder called `probmaps. The channel index you specified for inference is saved in the filename.

**References:**
S Saka, Y Wang, J Kishi, A Zhu, Y Zeng, W Xie, K Kirli, C Yapp, M Cicconet, BJ Beliveau, SW Lapan, S Yin, M Lin, E Boyde, PS Kaeser, G Pihan, GM Church, P Yin, Highly multiplexed in situ protein imaging with signal amplification by Immuno-SABER, Nat Biotechnology (accepted)

