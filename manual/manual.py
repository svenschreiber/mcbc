import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage

def load_images(path_template):
    images = []
    labels = []
    paths = glob.glob(path_template)
    for path in paths:
        label = path.split("\\")[-1]
        label = label.split("-")[0]
        labels.append(label)
        images.append(imread(path))
    return images, labels

trImgs, trLabels = load_images("../data/train/*.png")
teImgs, teLabels = load_images("../data/test/*.png")

trAvgs = [np.mean(trImgs, axis=(1,2)), trLabels]

def euk_dist(hist1, hist2):
    return np.sum((hist1-hist2)**2)**.5

classifications1 = []
for i in range(len(teImgs)):
    avg = np.mean(teImgs[i], axis=(0,1))
    dists = []
    for j in range(len(trImgs)):
        dist = euk_dist(avg, trAvgs[0][j])
        dists+=[[dist, trAvgs[1][j]]]
    dists.sort(key=lambda x: x[0])

    cDist, cLabel = dists[0]

    classifications1+=[cLabel]

correct_count = 0
for i in range(len(classifications1)):
    if classifications1[i] == teLabels[i]:
        correct_count += 1

print(classifications1)
print(teLabels)
print(correct_count / len(classifications1))