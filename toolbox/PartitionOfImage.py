import numpy as np
from toolbox.imtools import *
from toolbox.ftools import *

class PI2D:
    Image = None
    PaddedImage = None
    PatchSize = 128
    Margin = 14
    SubPatchSize = 100
    PC = None # patch coordinates
    NumPatches = 0
    Output = None
    Count = None
    NR = None
    NC = None
    NRPI = None
    NCPI = None
    Mode = None
    W = None

    def setup(image,patchSize,margin,mode):
        PI2D.Image = image
        PI2D.PatchSize = patchSize
        PI2D.Margin = margin
        subPatchSize = patchSize-2*margin
        PI2D.SubPatchSize = subPatchSize

        W = np.ones((patchSize,patchSize))
        W[[0,-1],:] = 0
        W[:,[0,-1]] = 0
        for i in range(1,2*margin):
            v = i/(2*margin)
            W[i,i:-i] = v
            W[-i-1,i:-i] = v
            W[i:-i,i] = v
            W[i:-i,-i-1] = v
        PI2D.W = W

        if len(image.shape) == 2:
            nr,nc = image.shape
        elif len(image.shape) == 3: # multi-channel image
            nz,nr,nc = image.shape

        PI2D.NR = nr
        PI2D.NC = nc

        npr = int(np.ceil(nr/subPatchSize)) # number of patch rows
        npc = int(np.ceil(nc/subPatchSize)) # number of patch cols

        nrpi = npr*subPatchSize+2*margin # number of rows in padded image 
        ncpi = npc*subPatchSize+2*margin # number of cols in padded image 

        PI2D.NRPI = nrpi
        PI2D.NCPI = ncpi

        if len(image.shape) == 2:
            PI2D.PaddedImage = np.zeros((nrpi,ncpi))
            PI2D.PaddedImage[margin:margin+nr,margin:margin+nc] = image
        elif len(image.shape) == 3:
            PI2D.PaddedImage = np.zeros((nz,nrpi,ncpi))
            PI2D.PaddedImage[:,margin:margin+nr,margin:margin+nc] = image

        PI2D.PC = [] # patch coordinates [r0,r1,c0,c1]
        for i in range(npr):
            r0 = i*subPatchSize
            r1 = r0+patchSize
            for j in range(npc):
                c0 = j*subPatchSize
                c1 = c0+patchSize
                PI2D.PC.append([r0,r1,c0,c1])

        PI2D.NumPatches = len(PI2D.PC)
        PI2D.Mode = mode # 'replace' or 'accumulate'

    def getPatch(i):
        r0,r1,c0,c1 = PI2D.PC[i]
        if len(PI2D.PaddedImage.shape) == 2:
            return PI2D.PaddedImage[r0:r1,c0:c1]
        if len(PI2D.PaddedImage.shape) == 3:
            return PI2D.PaddedImage[:,r0:r1,c0:c1]

    def createOutput():
        PI2D.Output = np.zeros(PI2D.PaddedImage.shape)
        if PI2D.Mode == 'accumulate':
            PI2D.Count = np.zeros((PI2D.NRPI,PI2D.NCPI))

    def patchOutput(i,P):
        r0,r1,c0,c1 = PI2D.PC[i]
        if PI2D.Mode == 'accumulate':
            PI2D.Count[r0:r1,c0:c1] += PI2D.W
        if len(P.shape) == 2:
            if PI2D.Mode == 'accumulate':
                PI2D.Output[r0:r1,c0:c1] += np.multiply(P,PI2D.W)
            elif PI2D.Mode == 'replace':
                PI2D.Output[r0:r1,c0:c1] = P
        elif len(P.shape) == 3:
            if PI2D.Mode == 'accumulate':
                PI2D.Output[:,r0:r1,c0:c1] += np.multiply(P,PI2D.W)
            elif PI2D.Mode == 'replace':
                PI2D.Output[:,r0:r1,c0:c1] = P

    def getValidOutput():
        margin = PI2D.Margin
        nr, nc = PI2D.NR, PI2D.NC
        if PI2D.Mode == 'accumulate':
            C = PI2D.Count[margin:margin+nr,margin:margin+nc]
        if len(PI2D.Output.shape) == 2:
            if PI2D.Mode == 'accumulate':
                return np.divide(PI2D.Output[margin:margin+nr,margin:margin+nc],C)
            if PI2D.Mode == 'replace':
                return PI2D.Output[margin:margin+nr,margin:margin+nc]
        if len(PI2D.Output.shape) == 3:
            if PI2D.Mode == 'accumulate':
                for i in range(PI2D.Output.shape[0]):
                    PI2D.Output[i,margin:margin+nr,margin:margin+nc] = np.divide(PI2D.Output[i,margin:margin+nr,margin:margin+nc],C)
            return PI2D.Output[:,margin:margin+nr,margin:margin+nc]


if __name__ == '__main__':
    I = np.random.rand(128,128)
    # PI2D.setup(I,128,14)
    PI2D.setup(I,64,4,'replace')

    PI2D.createOutput()

    for i in range(PI2D.NumPatches):
        P = PI2D.getPatch(i)
        PI2D.patchOutput(i,P)

    J = PI2D.getValidOutput()

    # I = I[0,:,:]
    # J = J[0,:,:]
    D = np.abs(I-J)
    print(np.max(D))

    K = cat(1,cat(1,I,J),D)
    imshow(K)