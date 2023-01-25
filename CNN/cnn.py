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
from keras.utils import image_dataset_from_directory
from keras.applications import EfficientNetV2B0


wkdir = os.path.dirname(os.path.realpath(__file__))
res_x, res_y = (384, 216)

timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
filename_pretrain = f"mcbc_weights_{timestamp}.h5"

train_set, val_set = image_dataset_from_directory(wkdir + "/../data/train/", batch_size=32, seed=1337, subset="both", validation_split=0.1, label_mode="categorical", image_size=(res_y, res_x))
test_set = image_dataset_from_directory(wkdir + "/../data/test/", batch_size=8, label_mode="categorical", image_size=(res_y, res_x))

es = EarlyStopping(monitor="val_loss", min_delta=0, patience=3)
mc = ModelCheckpoint(filepath=wkdir + f"/{filename_pretrain}", monitor="val_loss", save_best_only=True)

base_model = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(res_y, res_x, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


metrics = model.fit(train_set, validation_data=val_set, epochs=10, callbacks=[mc,es])
np.save(wkdir + f"/metric_{timestamp}.npy", metrics.history)

model.load_weights(wkdir + f"/{filename_pretrain}")

loss, acc = model.evaluate(test_set)
print(loss, acc)
