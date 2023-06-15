import os
import cv2
from os import listdir
from os.path import isfile, join

import random
import numpy as np

from imgaug import augmenters as iaa
import PIL.Image as Image
import matplotlib.pyplot as plt
# %matplotlib notebook 


covid_images_path = os.path.join("images", "COVID-19")
normal_images_path = os.path.join("images", "Normal")
bacteria_images_path = os.path.join("images", "Pneumonia")


covid_images = [os.path.join(covid_images_path, f) for f in listdir(covid_images_path) if isfile(join(covid_images_path, f))]
normal_images = [os.path.join(normal_images_path, f) for f in listdir(normal_images_path) if isfile(join(normal_images_path, f))]
bacteria_images = [os.path.join(bacteria_images_path, f) for f in listdir(bacteria_images_path) if isfile(join(bacteria_images_path, f))]


print(f"Total covid_images: {len(covid_images)}")
print(f"Total normal_images: {len(normal_images)}")
print(f"Total bacteria_images: {len(bacteria_images)}")


size=10


covid_images = random.sample(covid_images, size)
normal_images = random.sample(normal_images, size)
bacteria_images = random.sample(bacteria_images, size)


print(f"Total covid_images: {len(covid_images)}")
print(f"Total normal_images: {len(normal_images)}")
print(f"Total bacteria_images: {len(bacteria_images)}")


image_size = (100, 100)

def resize_image(img_array):
    img = Image.fromarray(img_array)
    img = img.resize(image_size)
    return np.array(img)

def show_images(images, title=""):
    fig, ax = plt.subplots(1, len(images), figsize=(15, 15), dpi=100)   
    for i, img in enumerate(images):
#         img = (img * 255).astype(np.uint8)
        ax[i].imshow(img)
        ax[i].set_title(title)
    [x.axis('off') for x in ax]
    plt.show()
    
def convert_images(image_paths):
    images = [np.array(Image.open(img).convert('RGB')) for img in image_paths]
    images = [resize_image(img) for img in images]    
    return np.array(images)

covid_image_arrs = convert_images(covid_images)
normal_image_arrs = convert_images(normal_images)
bacteria_image_arrs = convert_images(bacteria_images)


print(f"covid_image_arrs: {covid_image_arrs.shape}")
print(f"normal_image_arrs: {normal_image_arrs.shape}")
print(f"bacteria_image_arrs: {bacteria_image_arrs.shape}")


# Show random images that are Normal
show_images(random.sample(list(normal_image_arrs), 3), f"Normal {normal_image_arrs[0].shape}")

# Show random images that has COVID-19 Infection
show_images(random.sample(list(covid_image_arrs), 3), f"COVID-19 {covid_image_arrs[0].shape}")

# Show random images that has Bacteria Infection
show_images(random.sample(list(bacteria_image_arrs), 3), f"Bacteria {bacteria_image_arrs[0].shape}")


from sklearn.model_selection import train_test_split
import cv2

# 70:30 split on our dataset
train_size = int(len(covid_image_arrs) * 0.7)
test_size = len(covid_image_arrs) - train_size

# creating labels ["Normal", "COVID-19", "Bacteria"]
# labels = ["Normal", "COVID-19", "Bacteria", "Virus"]
labels = ["Normal", "COVID-19", "Bacteria"]
y1 = [[1., 0., 0.] for i in range(len(normal_image_arrs))]
y2 = [[0., 1., 0.] for i in range(len(covid_image_arrs))]
y3 = [[0., 0., 1.] for i in range(len(bacteria_image_arrs))]
# y4 = [[0., 0., 0., 1.] for i in range(len(virus_image_arrs))]

# splitting our data equally from each class
nX_train, nX_test, ny_train, ny_test = train_test_split(normal_image_arrs, y1, test_size=0.33, random_state=12)
cX_train, cX_test, cy_train, cy_test = train_test_split(covid_image_arrs, y2, test_size=0.33, random_state=12)
bX_train, bX_test, by_train, by_test = train_test_split(bacteria_image_arrs, y3, test_size=0.33, random_state=12)
# vX_train, vX_test, vy_train, vy_test = train_test_split(virus_image_arrs, y4, test_size=0.33, random_state=12)

# combining both of the classes 
# 4 Classes
# X_train = np.concatenate((cX_train, nX_train, bX_train, vX_train), axis=0)
# X_test = np.concatenate((cX_test, nX_test, bX_test, vX_test), axis=0)
# y_train = np.concatenate((cy_train, ny_train, by_train, vy_train), axis=0)
# y_test = np.concatenate((cy_test, ny_test, by_test, vy_test), axis=0)
# 3 Classes
X_train = np.concatenate((cX_train, nX_train, bX_train), axis=0)
X_test = np.concatenate((cX_test, nX_test, bX_test), axis=0)
y_train = np.concatenate((cy_train, ny_train, by_train), axis=0)
y_test = np.concatenate((cy_test, ny_test, by_test), axis=0)

# preparing our test data
shuffle_list = list(range(len(X_test)))
random.shuffle(shuffle_list)

shufX_test = []
shufy_test = []

for i in shuffle_list:
    shufX_test.append(X_test[i])
    shufy_test.append(y_test[i])

X_test = np.array(shufX_test)
y_test = np.array(shufy_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")



print(y_test[:5])



class data_generator:
    def create_train(self, train_set, test_set, batch_size, is_augmented=False):
        shape = train_set.shape[1:]
        while True:
            random_indexes = np.random.choice(len(train_set), batch_size)
            batch_images = []
            batch_labels = []
            for idx in random_indexes:
                # getting image
                image = train_set[idx]
                # augment image
                image = self.augment(image)
                # image historgram equalization
                image = self.histogram_equalization(image)
                # image denoise 
                # image = cv2.medianBlur(image, 3)
                # image normalization
                image = np.divide(image, 255)                
                batch_images.append(image)
                
                # getting label
                label = test_set[idx] 
                batch_labels.append(label)
            yield np.array(batch_images), np.array(batch_labels)
            
    def histogram_equalization(self, image):
        r,g,b = cv2.split(image)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        return np.stack((r,g,b), -1)
    
    def augment(self, image):  
        """
        Randomly process images to create more samples
        """
        sometimes = lambda aug: iaa.Sometimes(0.5, aug) # randomly apply 50% of the time
        augment_seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ]),
            sometimes(
            iaa.SomeOf((0,2),
            [
                iaa.GammaContrast((0.5, 2.0)),   
                iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                iaa.Affine(scale=(0.8, 1.2)),
                iaa.CoarseDropout(0.03, size_percent=0.05)
            ], random_order=True))
        ], random_order=True)
        
        image_aug = augment_seq.augment_image(image)
        return image_aug
    
    
# testing our data generator
batch_size = 45
generator = data_generator()
train_gen = generator.create_train(X_train, y_train, batch_size)
batch_img, batch_label = next(train_gen)

# visualise generate images
for l in batch_label[:4]:
    print(l, labels[np.argmax(l)])
    
show_images(batch_img[:4])




import tensorflow as tf

from tensorflow.keras import backend as K
#from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import Input, Dropout, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Flatten, Dense, MaxPooling2D, ReLU, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50V2, ResNet50, MobileNet, VGG16, DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import linear, softmax, sigmoid
#from tf_keras_vis.utils import utils
#from tf_keras_vis.visualization import visualize_cam

import warnings
warnings.filterwarnings('ignore')


# VGG16 transfer learning
def create_model(input_shape, n_out):
    model = Sequential()    
    model._name = "VGG16_Model"
    pretrain_model_1 = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape)) 
    pretrain_model_2 = MobileNet(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape)) 
    pretrain_model_3 = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape)) 
    pretrain_model_4 = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape)) 
    
    pretrain_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))    
    for layer in pretrain_model.layers: # Set all layers to be trainable
        layer.trainable = True
    for layer in pretrain_model.layers[-4:]: # last 4 layer freeze
        layer.trainable = False

    x = pretrain_model.output
    x = AveragePooling2D(pool_size=(3,3))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out, activation="softmax")(x)

    model = Model(pretrain_model.input, x)    
    return model


batch_size = 50
data_gen = data_generator()
train_gen = data_gen.create_train(X_train, y_train, batch_size)
test_gen = data_gen.create_train(X_test, y_test, batch_size)

opt = Adam(learning_rate=1e-4)
model_path = os.path.join("weights", "vgg16_covid19_weights.hd5")
checkPoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
earlyStop = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)

model = create_model(X_train.shape[1:], n_out=3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(f"Input shape: {X_train.shape[1:]}")
from keras.utils.vis_utils import plot_model

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit_generator(train_gen,
                             steps_per_epoch=len(X_train) // batch_size,
                             validation_data=next(test_gen),
                             validation_steps=len(y_test) // batch_size,
                             epochs=300,
                             verbose=1,
                             callbacks=[reduceLROnPlato, checkPoint, earlyStop])


# Draw learning curve
def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train accuracy")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()

show_history(history)


from keras.models import load_model
model = load_model('weights/vgg16_covid19_weights.hd5')



from sklearn.metrics import confusion_matrix, classification_report

predict_idxs = model.predict(X_test, batch_size=batch_size)
predict_idxs = np.argmax(predict_idxs, axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), predict_idxs)
print(cm)

print(classification_report(np.argmax(shufy_test, axis=1), predict_idxs, target_names=labels))