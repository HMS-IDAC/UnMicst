import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np
from skimage import io as skio
from scipy.ndimage import *
from skimage.morphology import *
from skimage.transform import resize

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

def imshowlist(L,**kwargs):
    n = len(L)
    for i in range(n):
        plt.subplot(1, n, i+1)
        if not kwargs:
            plt.imshow(L[i],cmap='gray')
        else:
            plt.imshow(L[i],**kwargs)
        plt.axis('off')
    plt.show()

def imread(path):
    return skio.imread(path)

def imwrite(I,path):
    skio.imsave(path,I)

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

def size(I):
    return list(I.shape)

def imresizeDouble(I,sizeOut): # input and output are double
    return resize(I,(sizeOut[0],sizeOut[1]),mode='reflect')

def imresize3Double(I,sizeOut): # input and output are double
    return resize(I,(sizeOut[0],sizeOut[1],sizeOut[2]),mode='reflect')

def imresizeUInt8(I,sizeOut): # input and output are UInt8
    return np.uint8(resize(I.astype(float),(sizeOut[0],sizeOut[1]),mode='reflect',order=0))

def imresize3UInt8(I,sizeOut): # input and output are UInt8
    return np.uint8(resize(I.astype(float),(sizeOut[0],sizeOut[1],sizeOut[2]),mode='reflect',order=0))

def normalize(I):
    m = np.min(I)
    M = np.max(I)
    if M > m:
        return (I-m)/(M-m)
    else:
        return I

def snormalize(I):
    m = np.mean(I)
    s = np.std(I)
    if s > 0:
        return (I-m)/s
    else:
        return I

def cat(a,I,J):
    return np.concatenate((I,J),axis=a)

def imerode(I,r):
    return binary_erosion(I, disk(r))

def imdilate(I,r):
    return binary_dilation(I, disk(r))

def imerode3(I,r):
    return morphology.binary_erosion(I, ball(r))

def imdilate3(I,r):
    return morphology.binary_dilation(I, ball(r))

def sphericalStructuralElement(imShape,fRadius):
    if len(imShape) == 2:
        return disk(fRadius,dtype=float)
    if len(imShape) == 3:
        return ball(fRadius,dtype=float)

def medfilt(I,filterRadius):
    return median_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def maxfilt(I,filterRadius):
    return maximum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def minfilt(I,filterRadius):
    return minimum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def ptlfilt(I,percentile,filterRadius):
    return percentile_filter(I,percentile,footprint=sphericalStructuralElement(I.shape,filterRadius))

def imgaussfilt(I,sigma,**kwargs):
    return gaussian_filter(I,sigma,**kwargs)

def imlogfilt(I,sigma,**kwargs):
    return -gaussian_laplace(I,sigma,**kwargs)

def imgradmag(I,sigma):
    if len(I.shape) == 2:
        dx = imgaussfilt(I,sigma,order=[0,1])
        dy = imgaussfilt(I,sigma,order=[1,0])
        return np.sqrt(dx**2+dy**2)
    if len(I.shape) == 3:
        dx = imgaussfilt(I,sigma,order=[0,0,1])
        dy = imgaussfilt(I,sigma,order=[0,1,0])
        dz = imgaussfilt(I,sigma,order=[1,0,0])
        return np.sqrt(dx**2+dy**2+dz**2)

def localstats(I,radius,justfeatnames=False):
    ptls = [10,30,50,70,90]
    featNames = []
    for i in range(len(ptls)):
        featNames.append('locPtl%d' % ptls[i])
    if justfeatnames == True:
        return featNames
    sI = size(I)
    nFeats = len(ptls)
    F = np.zeros((sI[0],sI[1],nFeats))
    for i in range(nFeats):
        F[:,:,i] = ptlfilt(I,ptls[i],radius)
    return F

def localstats3(I,radius,justfeatnames=False):
    ptls = [10,30,50,70,90]
    featNames = []
    for i in range(len(ptls)):
        featNames.append('locPtl%d' % ptls[i])
    if justfeatnames == True:
        return featNames
    sI = size(I)
    nFeats = len(ptls)
    F = np.zeros((sI[0],sI[1],sI[2],nFeats))
    for i in range(nFeats):
        F[:,:,:,i] = ptlfilt(I,ptls[i],radius)
    return F

def imderivatives(I,sigmas,justfeatnames=False):
    if type(sigmas) is not list:
        sigmas = [sigmas]
    derivPerSigmaFeatNames = ['d0','dx','dy','dxx','dxy','dyy','normGrad','normHessDiag']
    if justfeatnames == True:
        featNames = [];
        for i in range(len(sigmas)):
            for j in range(len(derivPerSigmaFeatNames)):
                featNames.append('derivSigma%d%s' % (sigmas[i],derivPerSigmaFeatNames[j]))
        return featNames
    nDerivativesPerSigma = len(derivPerSigmaFeatNames)
    nDerivatives = len(sigmas)*nDerivativesPerSigma
    sI = size(I)
    D = np.zeros((sI[0],sI[1],nDerivatives))
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        dx = imgaussfilt(I,sigma,order=[0,1])
        dy = imgaussfilt(I,sigma,order=[1,0])
        dxx = imgaussfilt(I,sigma,order=[0,2])
        dyy = imgaussfilt(I,sigma,order=[2,0])
        D[:,:,nDerivativesPerSigma*i  ] = imgaussfilt(I,sigma)
        D[:,:,nDerivativesPerSigma*i+1] = dx
        D[:,:,nDerivativesPerSigma*i+2] = dy
        D[:,:,nDerivativesPerSigma*i+3] = dxx
        D[:,:,nDerivativesPerSigma*i+4] = imgaussfilt(I,sigma,order=[1,1])
        D[:,:,nDerivativesPerSigma*i+5] = dyy
        D[:,:,nDerivativesPerSigma*i+6] = np.sqrt(dx**2+dy**2)
        D[:,:,nDerivativesPerSigma*i+7] = np.sqrt(dxx**2+dyy**2)
    return D
    # derivatives are indexed by the last dimension, which is good for ML features but not for visualization,
    # in which case the expected dimensions are [plane,channel,y(row),x(col)]; to obtain that ordering, do
    # D = np.moveaxis(D,[0,3,1,2],[0,1,2,3])

def imderivatives3(I,sigmas,justfeatnames=False):
    if type(sigmas) is not list:
        sigmas = [sigmas]

    derivPerSigmaFeatNames = ['d0','dx','dy','dz','dxx','dxy','dxz','dyy','dyz','dzz','normGrad','normHessDiag']

    # derivPerSigmaFeatNames = ['d0','normGrad','normHessDiag']

    if justfeatnames == True:
        featNames = [];
        for i in range(len(sigmas)):
            for j in range(len(derivPerSigmaFeatNames)):
                featNames.append('derivSigma%d%s' % (sigmas[i],derivPerSigmaFeatNames[j]))
        return featNames
    nDerivativesPerSigma = len(derivPerSigmaFeatNames)
    nDerivatives = len(sigmas)*nDerivativesPerSigma
    sI = size(I)
    D = np.zeros((sI[0],sI[1],sI[2],nDerivatives)) # plane, channel, y, x
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        dx  = imgaussfilt(I,sigma,order=[0,0,1]) # z, y, x
        dy  = imgaussfilt(I,sigma,order=[0,1,0])
        dz  = imgaussfilt(I,sigma,order=[1,0,0])
        dxx = imgaussfilt(I,sigma,order=[0,0,2])
        dyy = imgaussfilt(I,sigma,order=[0,2,0])
        dzz = imgaussfilt(I,sigma,order=[2,0,0])

        D[:,:,:,nDerivativesPerSigma*i   ] = imgaussfilt(I,sigma)
        D[:,:,:,nDerivativesPerSigma*i+1 ] = dx
        D[:,:,:,nDerivativesPerSigma*i+2 ] = dy
        D[:,:,:,nDerivativesPerSigma*i+3 ] = dz
        D[:,:,:,nDerivativesPerSigma*i+4 ] = dxx
        D[:,:,:,nDerivativesPerSigma*i+5 ] = imgaussfilt(I,sigma,order=[0,1,1])
        D[:,:,:,nDerivativesPerSigma*i+6 ] = imgaussfilt(I,sigma,order=[1,0,1])
        D[:,:,:,nDerivativesPerSigma*i+7 ] = dyy
        D[:,:,:,nDerivativesPerSigma*i+8 ] = imgaussfilt(I,sigma,order=[1,1,0])
        D[:,:,:,nDerivativesPerSigma*i+9 ] = dzz
        D[:,:,:,nDerivativesPerSigma*i+10] = np.sqrt(dx**2+dy**2+dz**2)
        D[:,:,:,nDerivativesPerSigma*i+11] = np.sqrt(dxx**2+dyy**2+dzz**2)

        # D[:,:,:,nDerivativesPerSigma*i   ] = imgaussfilt(I,sigma)
        # D[:,:,:,nDerivativesPerSigma*i+1 ] = np.sqrt(dx**2+dy**2+dz**2)
        # D[:,:,:,nDerivativesPerSigma*i+2 ] = np.sqrt(dxx**2+dyy**2+dzz**2)
    return D
    # derivatives are indexed by the last dimension, which is good for ML features but not for visualization,
    # in which case the expected dimensions are [plane,y(row),x(col)]; to obtain that ordering, do
    # D = np.moveaxis(D,[2,0,1],[0,1,2])

def imfeatures(I=[],sigmaDeriv=1,sigmaLoG=1,locStatsRad=0,justfeatnames=False):
    if type(sigmaDeriv) is not list:
        sigmaDeriv = [sigmaDeriv]
    if type(sigmaLoG) is not list:
        sigmaLoG = [sigmaLoG]
    derivFeatNames = imderivatives([],sigmaDeriv,justfeatnames=True)
    nLoGFeats = len(sigmaLoG)
    locStatsFeatNames = []
    if locStatsRad > 1:
        locStatsFeatNames = localstats([],locStatsRad,justfeatnames=True)
    nLocStatsFeats = len(locStatsFeatNames)
    if justfeatnames == True:
        featNames = derivFeatNames
        for i in range(nLoGFeats):
            featNames.append('logSigma%d' % sigmaLoG[i])
        for i in range(nLocStatsFeats):
            featNames.append(locStatsFeatNames[i])
        return featNames
    nDerivFeats = len(derivFeatNames)
    nFeatures = nDerivFeats+nLoGFeats+nLocStatsFeats
    sI = size(I)
    F = np.zeros((sI[0],sI[1],nFeatures))
    F[:,:,:nDerivFeats] = imderivatives(I,sigmaDeriv)
    for i in range(nLoGFeats):
        F[:,:,nDerivFeats+i] = imlogfilt(I,sigmaLoG[i])
    if locStatsRad > 1:
        F[:,:,nDerivFeats+nLoGFeats:] = localstats(I,locStatsRad)
    return F

def imfeatures3(I=[],sigmaDeriv=2,sigmaLoG=2,locStatsRad=0,justfeatnames=False):
    if type(sigmaDeriv) is not list:
        sigmaDeriv = [sigmaDeriv]
    if type(sigmaLoG) is not list:
        sigmaLoG = [sigmaLoG]
    derivFeatNames = imderivatives3([],sigmaDeriv,justfeatnames=True)
    nLoGFeats = len(sigmaLoG)
    locStatsFeatNames = []
    if locStatsRad > 1:
        locStatsFeatNames = localstats3([],locStatsRad,justfeatnames=True)
    nLocStatsFeats = len(locStatsFeatNames)
    if justfeatnames == True:
        featNames = derivFeatNames
        for i in range(nLoGFeats):
            featNames.append('logSigma%d' % sigmaLoG[i])
        for i in range(nLocStatsFeats):
            featNames.append(locStatsFeatNames[i])
        return featNames
    nDerivFeats = len(derivFeatNames)
    nFeatures = nDerivFeats+nLoGFeats+nLocStatsFeats
    sI = size(I)
    F = np.zeros((sI[0],sI[1],sI[2],nFeatures))
    F[:,:,:,:nDerivFeats] = imderivatives3(I,sigmaDeriv)
    for i in range(nLoGFeats):
        F[:,:,:,nDerivFeats+i] = imlogfilt(I,sigmaLoG[i])
    if locStatsRad > 1:
        F[:,:,:,nDerivFeats+nLoGFeats:] = localstats3(I,locStatsRad)
    return F

def stack2list(S):
    L = []
    for i in range(size(S)[2]):
        L.append(S[:,:,i])
    return L

def thrsegment(I,wsBlr,wsThr): # basic threshold segmentation
    G = imgaussfilt(I,sigma=(1-wsBlr)+wsBlr*5) # min 1, max 5
    M = G > wsThr
    return M