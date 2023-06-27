__all__ = ['gaussianFilter', 'files_to_array', 'formMontage', 'histogramMatching', 'create_tiles']

# Cell
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
from pathlib import Path
from PIL import Image
from skimage.transform import rescale
from skimage.exposure import match_histograms

# Cell

def gaussianFilter(numRowsInPatch,numColsInPatch):
    # Returns a 2D Gaussian kernel
    # You won't usually call this function directly.

    sigma = round(numRowsInPatch/4) # standard deviation of our Gaussian kernel.

    x = np.arange(numColsInPatch)
    y = np.arange(numRowsInPatch)
    kern1d_x = sp.stats.norm.pdf(x,round(numColsInPatch/2),sigma)
    kern1d_y = sp.stats.norm.pdf(y,round(numRowsInPatch/2),sigma)
    kern2d = np.outer(kern1d_y, kern1d_x)

    return kern2d/kern2d.sum()

# Cell
def files_to_array(dataDir,window_size,total_size,stride,suffix='_fakeB.tif'):
    assert window_size[0] == window_size[1], "Only square windows are allowed"
    numRowsInPatch = window_size[0]
    numColsInPatch = window_size[1]

    if window_size[0] != stride:
        numPatches_x = len(list(range(0,total_size[1]-window_size[1]+1,stride)))
        numPatches_y = len(list(range(0,total_size[0]-window_size[0]+1,stride)))
    else:
        numPatches_x = int(total_size[1]/window_size[0])
        numPatches_y = int(total_size[0]/window_size[1])


    images = np.zeros((numPatches_x,numPatches_y,numRowsInPatch,numColsInPatch,3))



    i = 0
    for blockRow in range(numPatches_x):
        for blockCol in range(numPatches_y):
            fname = os.path.join(dataDir, 'Stack' + str(i).zfill(4) +suffix)
            img = plt.imread(fname)
            images[blockRow,blockCol] = img

            i += 1

    return images


def formMontage(images, stride):
    # This is the function you use to create a montage image.
    # stride is a number such as 256.
    # If using 512 by 512 patches without overlapping, stride should be 512.

    # The next line tells you the shape of images:
    numPatches_x,numPatches_y,numRowsInPatch,numColsInPatch,numChannels = images.shape

    weights = gaussianFilter(numRowsInPatch,numColsInPatch)

    numRowsInMontage = numRowsInPatch + stride*(numPatches_x - 1)
    numColsInMontage = numColsInPatch + stride*(numPatches_y - 1)
    print(numRowsInMontage,numColsInMontage)

    montage = np.zeros((numRowsInMontage,numColsInMontage,3))
    montageWeights = np.zeros((numRowsInMontage,numColsInMontage))

    for i in range(numPatches_x):
        for j in range(numPatches_y):

            img = images[i,j]

            rowStart = stride*i
            rowStop = rowStart + numRowsInPatch
            colStart = stride*j
            colStop = colStart + numColsInPatch

            for channelIdx in range(numChannels):

                montage[rowStart:rowStop,colStart:colStop,channelIdx] += img[:,:,channelIdx]*weights

            montageWeights[rowStart:rowStop,colStart:colStop] += weights


    for channelIdx in range(numChannels):
        montage[:,:,channelIdx] = montage[:,:,channelIdx]/montageWeights

    return montage

# Cell

def histogramMatching(images, refImg):
    # If you want to perform histogram matching on each image in images,
    # you can use this function.
    # Each image in images will be matched with refImg.

    numPatches_x,numPatches_y,numRowsInPatch,numColsInPatch,numChannels = images.shape

    matchedImages = np.zeros(images.shape)
    for i in range(numPatches_y):
        for j in range(numPatches_x):

            print('Now matching patch in position (i,j) = (',i,',',j,')')

            img = images[i,j]
            matchedImg = match_histograms(img, refImg, multichannel=True)
            matchedImages[i,j] = matchedImg

    return matchedImages



# Cell
def create_tiles(filename,window_size,stride,new_folder):
    img = Image.open(filename)
    img = np.asarray(img)
    print(img.shape)
    if img.shape[-1]==4:
        img = img[:,:,:3]
    w = window_size[0]
    h = window_size[1]
    max_w = img.shape[0] - w
    max_h = img.shape[1] - h
    counter = 0
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for i in range(0,max_w+1,stride):
        for j in range(0,max_h+1,stride):
            tile = Image.fromarray(img[i:i+h,j:j+w])
            tile.save(new_folder+f'Stack{counter:04}.png')
            counter += 1
