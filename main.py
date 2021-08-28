from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initializing the learning rate

init_lr = 1e-4
epochs = 15
bs = 32

directory_train = r"P:\Folder Abra\Projects\Face mask detection-dataset\Dataset\train"
categories = ["with_mask", "without_mask"]

print("[INFO] LOADING IMAGES...")
data = []
labels = []

for category in categories:
    path = os.path.join(directory_train, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=base_model.input, outputs=head_model)

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# compile model
print("[INFO] COMPILING MODEL...")
opt = Adam(lr=init_lr, decay=init_lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] TRAINING HEAD...")
H = model.fit(
    aug.flow(train_x, train_y, batch_size=bs),
    steps_per_epoch=len(train_x)//bs,
    validation_data=(test_x, test_y),
    validation_steps=len(test_x)//bs,
    epochs=epochs)

# make predictions on the testing set
print("[INFO] EVALUATING NETWORK...")
pred_idx = model.predict(test_x, batch_size=bs)

# for each image in the testing set we need to find the index of the label with corresponding largest
# predicted probability
pred_idx = np.argmax(pred_idx, axis=1)

# show a nice formatted classification report
print(classification_report(test_y.argmax(axis=1), pred_idx, target_names=lb.classes_))

# serialize the model to disk
print("[INFO] SAVING MASK DETECTOR MODEL...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
n = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, n), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("LOSS/ACCURACY")
plt.legend(loc="lower left")
plt.savefig("plot.png")





