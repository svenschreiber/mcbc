import shutil
import numpy as np
import os

wkdir = os.path.dirname(os.path.realpath(__file__))

x_train = np.load(wkdir + "x_train_filenames.npy")
x_test = np.load(wkdir + "x_test_filenames.npy")

for file in x_train:
    label = file.split("-")[0]
    shutil.copyfile(wkdir + "all\\" + file, wkdir + f"train\\{label}\\{file}")

for file in x_test:
    label = file.split("-")[0]
    shutil.copyfile(wkdir + "all\\" + file, wkdir + f"test\\{label}\\{file}")