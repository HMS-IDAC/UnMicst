# **UnMicst - UNet Model for Identifying Cells and Segmenting Tissue**

# Image Preprocessing

Images can be preprocessed by inferring nuclei contours via a pretrained UNet model. The model is trained on 3 classes : background, nuclei contours and nuclei centers. The resulting probability maps can then be loaded into any modular segmentation pipeline that may use (but not limited to) a marker controlled watershed algorithm. 

The only **input** file is:
an .ome.tif or .tif  (preferably flat field corrected, minimal saturated pixels, and in focus. The model is trained on images acquired at 20x with binning 2x2 or a pixel size of 0.65 microns/px. If your settings differ, you can upsample/downsample to some extent.

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

**Running in a Conda environment**

If Docker is not available on your system, you can run the tool locally by creating a Conda environment. Ensure [conda](https://conda.io/en/latest/) is installed on your system, then clone the repo and use `conda.yml` to create the environment.

```
git clone https://github.com/HMS-IDAC/UnMicst.git
cd UnMicst
conda env create -f conda.yml
conda activate unmicst
python UnMicst.py /path/to/input.tif --outputPath /path/to/results/directory
```

**References:** <br/>
S Saka, Y Wang, J Kishi, A Zhu, Y Zeng, W Xie, K Kirli, C Yapp, M Cicconet, BJ Beliveau, SW Lapan, S Yin, M Lin, E Boyde, PS Kaeser, G Pihan, GM Church, P Yin, Highly multiplexed in situ protein imaging with signal amplification by Immuno-SABER, Nat Biotechnology (accepted)

