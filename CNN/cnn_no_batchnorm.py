import numpy as np
import os
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.metrics import Accuracy
from keras.metrics import Precision
from keras.metrics import CategoricalCrossentropy
from keras.utils import image_dataset_from_directory


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

model = Sequential()
model.add(Conv2D(32, (5, 5), padding="same", strides=1, input_shape=(res_y, res_x, 3)))
model.add(ReLU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), padding="same", strides=1))
model.add(ReLU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 5), padding="same", strides=1))
model.add(ReLU())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(ReLU())
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=[Accuracy(), Precision(), CategoricalCrossentropy()])

metrics = model.fit(train_set, validation_data=val_set, epochs=20, callbacks=[mc, es])
np.save(wkdir + f"/metric_{timestamp}.npy", metrics.history)