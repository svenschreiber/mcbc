import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage
import sklearn
from tqdm import tqdm

wkdir = os.path.dirname(os.path.realpath(__file__))

def dist(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2) ** .5

dataset_decrease_factor = 0.05 # => 8000 train imgs, 2000 test imgs
res_x = 384
res_y = 216

def load_images(dataset_type):
    images = []
    labels = []
    filenames = np.load(wkdir + f"/../data/x_{dataset_type}_filenames.npy")
    num_files = int(filenames.shape[0] * dataset_decrease_factor)
    for path in tqdm(filenames[:num_files]):
        label = path.split("-")[0]
        labels.append(label)
        images.append(imread(wkdir + f"/../data/{dataset_type}/{label}/{path}")[:,:,:3].reshape((res_x * res_y, 3)))
    return images, labels

trImgs, trLabels = load_images("train")
teImgs, teLabels = load_images("test")

binCount = 32

trHists = []
for i in tqdm(range(len(trImgs))):
    hist, _ = np.histogramdd(trImgs[i], bins=binCount)
    trHists += [hist]

correctCount = 0

classifications = []
for i in tqdm(range(len(teImgs))):
    tsHist, _ = np.histogramdd(teImgs[i], bins=binCount)
    dists = []
    for j in range(len(trHists)):
        distance = dist(tsHist, trHists[j])
        dists += [[distance, trLabels[j]]]
    dists.sort(key=lambda x: x[0])
    # Nächster Nachbar
    cDist, cLabel = dists[0]

    classifications += [cLabel]

    if cLabel == teLabels[i]:
        correctCount += 1

print("Accuracy:", correctCount / len(classifications))
