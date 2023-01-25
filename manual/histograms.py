import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage
import sklearn

wkdir = os.path.dirname(os.path.realpath(__file__))

def dist(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2) ** .5

dataset_decrease_factor = 0.05 # => 8000 train imgs, 2000 test imgs
res_y = 216

def load_images(dataset_type):
    images = []
    labels = []
    filenames = np.load(wkdir + f"/../data/x_{dataset_type}_filenames.npy")
    num_files = int(filenames.shape[0] * dataset_decrease_factor)
    for path in filenames[:num_files]:
        label = path.split("-")[0]
        labels.append(label)
        images.append(imread(wkdir + f"/../data/{dataset_type}/{label}/{path}")[res_y//2:])
    return images, labels


trImgs, trLabels = load_images("train")
teImgs, teLabels = load_images("test")

trAvgs = [np.mean(trImgs, axis=(1, 2)), trLabels]

binCount = 54

def combinedHist(img):
    histR, _ = np.histogram(img[:, :, 0].flatten(), bins=binCount, density=True)
    histG, _ = np.histogram(img[:, :, 1].flatten(), bins=binCount, density=True)
    histB, _ = np.histogram(img[:, :, 2].flatten(), bins=binCount, density=True)
    return np.hstack((histR, histG, histB))


trHists = []
for i in range(len(trImgs)):
    trHists += [[combinedHist(trImgs[i]), trLabels[i]]]

correctCount = 0

classifications = []
for i in range(len(teImgs)):
    tsHist = combinedHist(teImgs[i])
    dists = []
    for j in range(len(trHists)):
        distance = dist(tsHist, trHists[j][0])
        dists += [[distance, trHists[j][1]]]
    dists.sort(key=lambda x: x[0])
    # Nächster Nachbar
    cDist, cLabel = dists[0]

    classifications += [cLabel]

    if cLabel == teLabels[i]:
        correctCount += 1

print("Accuracy:", correctCount / len(classifications))
