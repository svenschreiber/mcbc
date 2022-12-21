import numpy as np
import glob
from skimage.io import imread
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib


lables_dict = {"aquatic": 0, "snowy": 1, "arid": 2, "forest": 3, "plains": 4}


def load_images(path_template):
    paths = glob.glob(path_template)
    labels = []
    images = np.zeros((len(paths), 360, 640, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        label = path.split("\\")[-1]
        label = label.split("-")[0]
        labels.append(lables_dict[label])
        images[i] = imread(path)[:, :, :3]
    return images, labels


trImgs, trLabels = load_images("..\\data\\train\\*.png")
tsImgs, tsLabels = load_images("..\\data\\test\\*.png")
trLabels = to_categorical(trLabels, 5)
tsLabels = to_categorical(tsLabels, 5)

filename_pretrain = "minecraft_biomes.h5"

es = EarlyStopping(monitor="val_loss", min_delta=0, patience=3)
mc = ModelCheckpoint(filepath=filename_pretrain, monitor="val_loss", save_best_only=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(360, 640, 3), name="conv1"))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(trImgs, trLabels, batch_size=16, epochs=20, validation_split=0.2, callbacks=[es, mc])

# model.load_weights(filename_pretrain)
loss, acc = model.evaluate(tsImgs, tsLabels)
print(loss, acc)
import numpy as np
import glob
from skimage.io import imread
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib


print(device_lib.list_local_devices())

lables_dict = {"aquatic": 0, "snowy": 1, "arid": 2, "forest": 3, "plains": 4}


def load_images(path_template):
    paths = glob.glob(path_template)
    labels = []
    images = np.zeros((len(paths), 360, 640, 4), dtype=np.uint8)
    for i, path in enumerate(paths):
        label = path.split("\\")[-1]
        label = label.split("-")[0]
        labels.append(lables_dict[label])
        images[i] = imread(path)
    return images, labels


x_train, trLabels = load_images("..\\data\\train\\*.png")
x_test, teLabels = load_images("..\\data\\test\\*.png")
y_train = to_categorical(trLabels, 5)
y_test = to_categorical(teLabels, 5)

# es = EarlyStopping(monitor="val_loss", min_delta=0, patience=3)
# mc = ModelCheckpoint(filepath=filename_pretrain, monitor="val_loss", save_best_only=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(360, 640, 4), name="conv1"))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=20, validation_split=0.2)  # callbacks=[es, mc]

# model.load_weights(filename_pretrain)
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)
