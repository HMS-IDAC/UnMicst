# UnMICST - Universal Models for Identifying Cells and Segmenting Tissue <br>
![](/images/unmicstbannerv2.png) <br>
<p align="center"> 
  (pronounced un-mixed)
</p>

## Introduction
Nuclei segmentation, especially for tissues, is a challenging and unsolved problem. Convolutional neural networks are particularly well-suited for this task: separating the foreground class (nuclei pixels) from the background class. UnMICST generates probability maps where the intensity at each pixel defines how confident the pixel has been correctly classified to the aforementioned classes. These maps can make downstream image binarization more accurate using tools such as s3segmenter. https://github.com/HMS-IDAC/S3segmenter. UnMICST currently uses the UNet architecture (Ronneberger et al., 2015) but Mask R-CNN and Pyramid Scene Parsing (PSP)Net are coming very soon! **The concept, models, and training data are featured here: https://www.biorxiv.org/content/10.1101/2021.04.02.438285v1 **

![](/images/probmaps.png)
## Training data / annotations
![](/images/TMAv2.png)
**Training data can be found here: https://www.synapse.org/#!Synapse:syn24192218/files/ and includes:**
- training images from 7 tissue types that appeared to encapsulate the different morphologies of the entire tissue microarray: 1) lung adenocarcinoma, 2) non-neoplastic prostate, 3) non-neoplastic small intestine, 4) non-neoplastic ovary, 5) tonsil, 6) glioblastoma, and 7) colon adenocarcinoma. 
- manual annotations of the nuclei centers, contours, and background of the abovementioned tissue types<br>
- DNA channel and nuclear envelope staining (lamin B and nucleoporin 98) for improved accuracy<br>


![](/images/annotationsmontagev2.png)<br>
*nuclear envelope stain (left), DNA stain (center), manual annotations for nuclei contours, centers and background (right)*<br>

- intentionally defocused planes and saturated pixels for better dealing with real-world artefacts<br>
![](/images/realaugmentations.png)<br>
*additional z-planes above and below the focal plane were acquired in widefield microscopy and included in the training as part of "real" augmentation as opposed to Gaussian blurring, which is less effective*<br>

**The training data is publicly available under creative commons license for expansion or training newer models.**


# Prerequisite files
-an .ome.tif or .tif  (preferably flat field corrected, minimal saturated pixels, and in focus. The model is trained on images acquired at a pixelsize of 0.65 microns/px. If your settings differ, you can upsample/downsample to some extent.

# Expected output files
1. a tiff stack where the different probability maps for each class are concatenated in the Z-axis in the order: nuclei foreground, nuclei contours, and background with suffix *_Probabilities*
2. a QC image with the DNA image concatenated with the nuclei contour probability map with suffix *_Preview*

# Parameter list
1. `--tool` : specify which UnMICST version you want to use (ie. UnMicst, UnMicst1-5, UnMicst2). v1 is deprecated. v1.5 uses the DNA channel only. v2 uses DNA and nuclear envelope staining.
2. `--channel` : specify the channel(s) to be used. 
3. `--scalingFactor` : an upsample or downsample factor if your pixel sizes are mismatched from the dataset.
4. `--mean` and `--std` : If your image is vastly different in terms of brightness/contrast, enter the image mean and standard deviation here.


**Running as a Docker container**

The docker image is distributed through Dockerhub and includes `UnMicst` with all of its dependencies. Use `docker pull` to retrieve a specific version or the `latest` tag:

```
docker pull labsyspharm/unmicst:latest
docker pull labsyspharm/unmicst:2.6.11
```

Instatiate a container and mount the input directory containing your image.
```
docker run -it --runtime=nvidia -v /path/to/data:/data labsyspharm/unmicst:latest bash
```
When running the code without using a GPU, `--runtime=nvidia` can be omitted:
```
docker run -it -v /path/to/data:/data labsyspharm/unmicst:latest bash
```

UnMicst resides in the `/app` directory inside the container:

```
root@0ea0cdc46c8f:/# python app/UnMicst.py /data/input/my.tif --outputPath /data/results
```


**References:** <br/>
Clarence Yapp, Edward Novikov, Won-Dong Jang, Yu-An Chen, Marcelo Cicconet, Zoltan Maliga, Connor A. Jacobson, Donglai Wei, Sandro Santagata, Hanspeter Pfister, Peter K. Sorger, 2021, UnMICST: Deep learning with real augmentation for robust segmentation of highly multiplexed images of human tissues

S Saka, Y Wang, J Kishi, A Zhu, Y Zeng, W Xie, K Kirli, C Yapp, M Cicconet, BJ Beliveau, SW Lapan, S Yin, M Lin, E Boyde, PS Kaeser, G Pihan, GM Church, P Yin, 2020, Highly multiplexed in situ protein imaging with signal amplification by Immuno-SABER, Nat Biotechnology 

