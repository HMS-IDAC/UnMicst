# UnMICST - UNet Model for Identifying Cells and Segmenting Tissue <br>
*(pronounced un-mixed)* <br>

## Introduction
Nuclei segmentation, especially for tissues, is a challenging and unsolved problem. Convolutional neural networks are particularly well-suited for this task - classifying image pixels into nuclei centers, nuclei contours and background. UnMICST generates probability maps where the intensity at each pixel defines how confident the pixel has been correctly classified to the aforementioned classes. These maps that can make downstream image binarization more accurate using tools such as s3segmenter. https://github.com/HMS-IDAC/S3segmenter. UnMICST currently uses the UNet architecture but Mask R-CNN and Pyramid Scene Parsing (PSP)Net are coming very soon!
![](/images/probmaps.png)
## Training data / annotations
![](/images/TMAv2.png)
**Training data can be found here: https://www.synapse.org/#!Synapse:syn24192218/files/ and includes:**
- training images from 7 tissue types that appeared to encapsulate the different morphologies of the entire tissue microarray: 1) lung adenocarcinoma, 2) non-neoplastic prostate, 3) non-neoplastic small intestine, 4) non-neoplastic ovary, 5) tonsil, 6) glioblastoma, and 7) colon adenocarcinoma. 
- manual annotations of the nuclei centers, contours, and background of the abovementioned tissue types<br>
- DNA channel and nuclear envelope staining (lamin B and nucleoporin 98) for improved accuracy<br>

![](/images/annotationsmontagev2.png)<br>

- intentionally defocused planes and saturated pixels for better dealing with real-world artefacts<br>

![](/images/realaugmentations.png)<br>
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

The docker image is distributed through Dockerhub and includes `UnMicst` with all of its dependencies. Parallel images with and without gpu support are available.

```
docker pull labsyspharm/unmicst:latest
docker pull labsyspharm/unmicst:latest-gpu
```

Instatiate a container and mount the input directory containing your image.
```
docker run -it --runtime=nvidia -v /path/to/data:/data labsyspharm/unmicst:latest-gpu bash
```
When using the CPU-only image, `--runtime=nvidia` can be omitted:
```
docker run -it -v /path/to/data:/data labsyspharm/unmicst:latest bash
```

UnMicst resides in the `/app` directory inside the container:

```
root@0ea0cdc46c8f:/# python app/UnMicst.py /data/input/my.tif --outputPath /data/results
```


**References:** <br/>
S Saka, Y Wang, J Kishi, A Zhu, Y Zeng, W Xie, K Kirli, C Yapp, M Cicconet, BJ Beliveau, SW Lapan, S Yin, M Lin, E Boyde, PS Kaeser, G Pihan, GM Church, P Yin, Highly multiplexed in situ protein imaging with signal amplification by Immuno-SABER, Nat Biotechnology (accepted)

