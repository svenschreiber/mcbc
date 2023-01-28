import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage
from collections import Counter
from tqdm import tqdm

wkdir = os.path.dirname(os.path.realpath(__file__))

dataset_decrease_factor = 0.05 # => 8000 train imgs, 2000 test imgs
res_y = 216

def load_images(dataset_type):
    images = []
    labels = []
    filenames = np.load(wkdir + f"/../data/x_{dataset_type}_filenames.npy")
    num_files = int(filenames.shape[0] * dataset_decrease_factor)
    for path in tqdm(filenames[:num_files]):
        label = path.split("-")[0]
        labels.append(label)
        images.append(imread(wkdir + f"/../data/{dataset_type}/{label}/{path}")[res_y//2:])
    return images, labels


trImgs, trLabels = load_images("train")
teImgs, teLabels = load_images("test")

trAvgs = [np.mean(trImgs, axis=(1, 2)), trLabels]

def euk_dist(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2) ** .5

classifications = []
for i in tqdm(range(len(teImgs))):
    avg = np.mean(teImgs[i], axis=(0, 1))
    dists = []
    for j in range(len(trImgs)):
        dist = euk_dist(avg, trAvgs[0][j])
        dists += [[dist, trAvgs[1][j]]]
    dists.sort(key=lambda x: x[0])

    cDist, cLabel = dists[0]

    classifications += [cLabel]

correct_count = 0
for i in range(len(classifications)):
    if classifications[i] == teLabels[i]:
        correct_count += 1

print("Accuracy (euclidean distance):", correct_count / len(classifications))
