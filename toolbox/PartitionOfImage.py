import numpy as np
from toolbox.imtools import *
# from toolbox.ftools import *
# import sys

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

    def createOutput(nChannels):
        if nChannels == 1:
            PI2D.Output = np.zeros((PI2D.NRPI,PI2D.NCPI),np.float16)
        else:
            PI2D.Output = np.zeros((nChannels,PI2D.NRPI,PI2D.NCPI),np.float16)
        if PI2D.Mode == 'accumulate':
            PI2D.Count = np.zeros((PI2D.NRPI,PI2D.NCPI),np.float16)

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
                for i in range(P.shape[0]):
                    PI2D.Output[i,r0:r1,c0:c1] += np.multiply(P[i,:,:],PI2D.W)
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


    def demo():
        I = np.random.rand(128,128)
        # PI2D.setup(I,128,14)
        PI2D.setup(I,64,4,'replace')

        nChannels = 2
        PI2D.createOutput(nChannels)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            Q = np.zeros((nChannels,P.shape[0],P.shape[1]))
            for j in range(nChannels):
                Q[j,:,:] = P
            PI2D.patchOutput(i,Q)

        J = PI2D.getValidOutput()
        J = J[0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        K = cat(1,cat(1,I,J),D)
        imshow(K)


class PI3D:
    Image = None
    PaddedImage = None
    PatchSize = 128
    Margin = 14
    SubPatchSize = 100
    PC = None # patch coordinates
    NumPatches = 0
    Output = None
    Count = None
    NR = None # rows
    NC = None # cols
    NZ = None # planes
    NRPI = None
    NCPI = None
    NZPI = None
    Mode = None
    W = None

    def setup(image,patchSize,margin,mode):
        PI3D.Image = image
        PI3D.PatchSize = patchSize
        PI3D.Margin = margin
        subPatchSize = patchSize-2*margin
        PI3D.SubPatchSize = subPatchSize

        W = np.ones((patchSize,patchSize,patchSize))
        W[[0,-1],:,:] = 0
        W[:,[0,-1],:] = 0
        W[:,:,[0,-1]] = 0
        for i in range(1,2*margin):
            v = i/(2*margin)
            W[[i,-i-1],i:-i,i:-i] = v
            W[i:-i,[i,-i-1],i:-i] = v
            W[i:-i,i:-i,[i,-i-1]] = v

        PI3D.W = W

        if len(image.shape) == 3:
            nz,nr,nc = image.shape
        elif len(image.shape) == 4: # multi-channel image
            nz,nw,nr,nc = image.shape

        PI3D.NR = nr
        PI3D.NC = nc
        PI3D.NZ = nz

        npr = int(np.ceil(nr/subPatchSize)) # number of patch rows
        npc = int(np.ceil(nc/subPatchSize)) # number of patch cols
        npz = int(np.ceil(nz/subPatchSize)) # number of patch planes

        nrpi = npr*subPatchSize+2*margin # number of rows in padded image 
        ncpi = npc*subPatchSize+2*margin # number of cols in padded image 
        nzpi = npz*subPatchSize+2*margin # number of plns in padded image 

        PI3D.NRPI = nrpi
        PI3D.NCPI = ncpi
        PI3D.NZPI = nzpi

        if len(image.shape) == 3:
            PI3D.PaddedImage = np.zeros((nzpi,nrpi,ncpi))
            PI3D.PaddedImage[margin:margin+nz,margin:margin+nr,margin:margin+nc] = image
        elif len(image.shape) == 4:
            PI3D.PaddedImage = np.zeros((nzpi,nw,nrpi,ncpi))
            PI3D.PaddedImage[margin:margin+nz,:,margin:margin+nr,margin:margin+nc] = image

        PI3D.PC = [] # patch coordinates [z0,z1,r0,r1,c0,c1]
        for iZ in range(npz):
            z0 = iZ*subPatchSize
            z1 = z0+patchSize
            for i in range(npr):
                r0 = i*subPatchSize
                r1 = r0+patchSize
                for j in range(npc):
                    c0 = j*subPatchSize
                    c1 = c0+patchSize
                    PI3D.PC.append([z0,z1,r0,r1,c0,c1])

        PI3D.NumPatches = len(PI3D.PC)
        PI3D.Mode = mode # 'replace' or 'accumulate'

    def getPatch(i):
        z0,z1,r0,r1,c0,c1 = PI3D.PC[i]
        if len(PI3D.PaddedImage.shape) == 3:
            return PI3D.PaddedImage[z0:z1,r0:r1,c0:c1]
        if len(PI3D.PaddedImage.shape) == 4:
            return PI3D.PaddedImage[z0:z1,:,r0:r1,c0:c1]

    def createOutput(nChannels):
        if nChannels == 1:
            PI3D.Output = np.zeros((PI3D.NZPI,PI3D.NRPI,PI3D.NCPI))
        else:
            PI3D.Output = np.zeros((PI3D.NZPI,nChannels,PI3D.NRPI,PI3D.NCPI))
        if PI3D.Mode == 'accumulate':
            PI3D.Count = np.zeros((PI3D.NZPI,PI3D.NRPI,PI3D.NCPI))

    def patchOutput(i,P):
        z0,z1,r0,r1,c0,c1 = PI3D.PC[i]
        if PI3D.Mode == 'accumulate':
            PI3D.Count[z0:z1,r0:r1,c0:c1] += PI3D.W
        if len(P.shape) == 3:
            if PI3D.Mode == 'accumulate':
                PI3D.Output[z0:z1,r0:r1,c0:c1] += np.multiply(P,PI3D.W)
            elif PI3D.Mode == 'replace':
                PI3D.Output[z0:z1,r0:r1,c0:c1] = P
        elif len(P.shape) == 4:
            if PI3D.Mode == 'accumulate':
                for i in range(P.shape[1]):
                    PI3D.Output[z0:z1,i,r0:r1,c0:c1] += np.multiply(P[:,i,:,:],PI3D.W)
            elif PI3D.Mode == 'replace':
                PI3D.Output[z0:z1,:,r0:r1,c0:c1] = P

    def getValidOutput():
        margin = PI3D.Margin
        nz, nr, nc = PI3D.NZ, PI3D.NR, PI3D.NC
        if PI3D.Mode == 'accumulate':
            C = PI3D.Count[margin:margin+nz,margin:margin+nr,margin:margin+nc]
        if len(PI3D.Output.shape) == 3:
            if PI3D.Mode == 'accumulate':
                return np.divide(PI3D.Output[margin:margin+nz,margin:margin+nr,margin:margin+nc],C)
            if PI3D.Mode == 'replace':
                return PI3D.Output[margin:margin+nz,margin:margin+nr,margin:margin+nc]
        if len(PI3D.Output.shape) == 4:
            if PI3D.Mode == 'accumulate':
                for i in range(PI3D.Output.shape[1]):
                    PI3D.Output[margin:margin+nz,i,margin:margin+nr,margin:margin+nc] = np.divide(PI3D.Output[margin:margin+nz,i,margin:margin+nr,margin:margin+nc],C)
            return PI3D.Output[margin:margin+nz,:,margin:margin+nr,margin:margin+nc]


    def demo():
        I = np.random.rand(128,128,128)
        PI3D.setup(I,64,4,'accumulate')

        nChannels = 2
        PI3D.createOutput(nChannels)

        for i in range(PI3D.NumPatches):
            P = PI3D.getPatch(i)
            Q = np.zeros((P.shape[0],nChannels,P.shape[1],P.shape[2]))
            for j in range(nChannels):
                Q[:,j,:,:] = P
            PI3D.patchOutput(i,Q)

        J = PI3D.getValidOutput()
        J = J[:,0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        pI = I[64,:,:]
        pJ = J[64,:,:]
        pD = D[64,:,:]

        K = cat(1,cat(1,pI,pJ),pD)
        imshow(K)

