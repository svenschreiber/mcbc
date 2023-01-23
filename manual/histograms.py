import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage

individualOutput = False


def dist(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2) ** .5


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

print(classifications)
print(teLabels)
print(correctCount / len(classifications))
