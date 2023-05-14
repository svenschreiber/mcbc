import numpy as np
import os
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.metrics import Accuracy
from keras.metrics import Precision
from keras.metrics import CategoricalCrossentropy
from keras.utils import image_dataset_from_directory
from keras.applications import EfficientNetV2B2
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


wkdir = os.path.dirname(os.path.realpath(__file__))
res_x, res_y = (384, 216)

timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
filename_pretrain = f"mcbc_weights_2023_01_27-01_06_37.h5"

train_set, val_set = image_dataset_from_directory(wkdir + "/../data/train/", batch_size=32, seed=1337, subset="both",
                                                  validation_split=0.1, label_mode="categorical",
                                                  image_size=(res_y, res_x))
test_set = image_dataset_from_directory(wkdir + "/../data/test/", batch_size=32, shuffle=False, seed=1337, label_mode="categorical",
                                        image_size=(res_y, res_x))
es = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
mc = ModelCheckpoint(filepath=wkdir + f"/{filename_pretrain}", monitor="val_loss", save_best_only=True)

base_model = EfficientNetV2B2(include_top=False, weights="imagenet", input_shape=(res_y, res_x, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=[Accuracy(), Precision(), CategoricalCrossentropy()])

metrics = model.fit(train_set, validation_data=val_set, epochs=20, callbacks=[mc, es])
np.save(wkdir + f"/metric_{timestamp}.npy", metrics.history)

# Confusion matrix generation
""""
model.load_weights(wkdir + f"/{filename_pretrain}")
y_pred = model.predict(test_set)
print(y_pred.argmax(axis=1))
y_test = np.concatenate([y for _, y in test_set.as_numpy_iterator()], axis=0)
conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(conf_mat)
labels = ["aquatic", "arid", "forest", "plains", "snowy"]
df_cm = pd.DataFrame(conf_mat, labels, labels)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues') # font size

plt.show()
"""