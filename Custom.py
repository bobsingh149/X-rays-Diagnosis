import os
from os import path, listdir
from os.path import join, isfile
from glob import glob
from pathlib import Path
import cv2

import random
import numpy as np
import pandas as pd
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import PIL.Image as Image
import matplotlib.pyplot as plt

covid_images_path = os.path.join("images", "COVID-19")
normal_images_path = os.path.join("images", "Normal")
bacteria_images_path = os.path.join("images", "Pneumonia")

covid_images = [os.path.join(covid_images_path, f) for f in listdir(covid_images_path) if
                isfile(join(covid_images_path, f))]
normal_images = [os.path.join(normal_images_path, f) for f in listdir(normal_images_path) if
                 isfile(join(normal_images_path, f))]
bacteria_images = [os.path.join(bacteria_images_path, f) for f in listdir(bacteria_images_path) if
                   isfile(join(bacteria_images_path, f))]

print(f"Total covid_images: {len(covid_images)}")
print(f"Total normal_images: {len(normal_images)}")
print(f"Total bacteria_images: {len(bacteria_images)}")

size = 188

covid_images = random.sample(covid_images, size)
normal_images = random.sample(normal_images, size)
bacteria_images = random.sample(bacteria_images, size)

print(f"Total covid_images: {len(covid_images)}")
print(f"Total normal_images: {len(normal_images)}")
print(f"Total bacteria_images: {len(bacteria_images)}")

dataset = []
for file in covid_images:
    dataset.append([Path(covid_images[0]).parent.name, file])

for file in normal_images:
    dataset.append([Path(normal_images[0]).parent.name, file])

for file in bacteria_images:
    dataset.append([Path(bacteria_images[0]).parent.name, file])

# for file in virus_images:
#     dataset.append([Path(virus_images[0]).parent.name, file])

pd.set_option('max_colwidth', 500)
df = pd.DataFrame(dataset)
df.columns = ['Class', 'Path']
total_labels = len(set(df['Class'].values))
labels = set(df['Class'].values)

print(f"Total number of labels: {total_labels}")
print(df.info())

df.sample(n=5)

image_shape = (100, 100, 3)


def resize_image(img_array):
    img = Image.fromarray(img_array)
    img = img.resize(image_shape[:-1])
    return np.array(img)


def show_images(images, title=""):
    fig, ax = plt.subplots(1, len(images), figsize=(10, 10), dpi=100)
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].set_title(title)
    [x.axis('off') for x in ax]
    plt.show()


def load_images(image_paths):
    images = [np.array(Image.open(img).convert('RGB')) for img in image_paths]
    images = [resize_image(img) for img in images]
    return np.array(images)


# View random 3 samples from each character label
for l in labels:
    char_imgs = df[df['Class'] == l]['Path']
    char_imgs = load_images(char_imgs.values)
    show_images(random.sample(list(char_imgs), 3), l)

train_list = []
test_list = []

for l in labels:
    char_imgs = df[df['Class'] == l]

    print(l)
    df_train, df_test = train_test_split(char_imgs, test_size=0.85, random_state=42)

    train_list.append(df_train)
    test_list.append(df_test)

df_X_train = pd.concat(train_list)
df_X_test = pd.concat(test_list)

# Train samples
df_X_train.groupby('Class').count()

# Test samples
df_X_test.groupby('Class').count()


class Data_Generator(object):

    def __init__(self, df, image_shape, batch_size):
        # Prepare parameters
        self.df = df.copy()
        self.h, self.w, self.c = image_shape
        self.batch_size = batch_size
        self.labels = list(set(self.df['Class']))

    def resize_image(self, img_array):
        img = Image.fromarray(img_array)
        img = img.resize(image_shape[:-1])
        return np.array(img)

    def load_image(self, url):
        img = Image.open(url).convert('RGB')
        img = np.array(img)
        img = self.resize_image(img)
        return img

    '''import pandas as pd

    df = pd.DataFrame({'c1': [10, 11, 12], 'c2': [100, 110, 120]})
    df = df.reset_index()  # make sure indexes pair with number of rows

    for index, row in df.iterrows():
        print(row['c1'], row['c2'])'''

    def get_hard_pair(self,df1,df2,same=True):

        n1 = len(df1.index)
        n2=len(df2.index)

        size = n1*n2
        temp_pairs = [np.zeros((size, self.h, self.w, self.c)) for i in range(2)]

        idx = 0
        for r1i,row1 in df1.iterrows():

            for r2i,row2 in df2.iterrows():
                image1 = row1['Path'].values[0]
                image1 = self.load_image(image1)
                image1 = self.histogram_equalization(image1)

                image2 =  row2['Path'].values[0]
                image2 = self.load_image(image2)
                image2 = self.histogram_equalization(image2)

                if idx>=size:
                    break
                temp_pairs[0][idx, :, :, :] = image1
                temp_pairs[1][idx, :, :, :] = image2

                idx+=1



        model= load_model(join("weights", model_name))


        pred = model.predict(temp_pairs)

        hard_idx=0

        if same:
            hard_idx=np.argmin(pred)[0]
        else:
            hard_idx=np.argmax(pred)[0]


        hard_img1=temp_pairs[0][hard_idx]
        hard_img2=temp_pairs[1][hard_idx]


        return hard_img1,hard_img2


    def histogram_equalization(self, image):
        r, g, b = cv2.split(image)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        img = np.stack((r, g, b), -1)
        img = np.divide(img, 255)
        return img

    def get_batch(self):

        while True:
            # Create holder for batches
            pairs = [np.zeros((self.batch_size, self.h, self.w, self.c)) for i in range(2)]
            targets = np.zeros((self.batch_size,))
            targets[self.batch_size // 2:] = 1  # half are positive half are negative
            random.shuffle(targets)

            rand_num=random.randint(1,31)

            if rand_num%5 == 0:

                for b in range(self.batch_size):
                    # Select anchor image
                    selected_label = np.random.choice(self.labels, 1)[0]

                    # Negative - 0 (Different images), Positive = 1 (Same images)
                    if targets[b] == 0:
                        # Negative examples
                        labels_ = self.labels.copy()
                        labels_.remove(selected_label)
                        target_label = np.random.choice(labels_, 1, replace=False)[0]



                        df1=self.df[self.df["Class"] == selected_label]
                        df2=self.df[self.df["Class"] == target_label]
                        image1,image2=self.get_hard_pair(df1=df1,df2=df2,same=False)
                        pairs[0][b, :, :, :] = image1
                        pairs[1][b, :, :, :] = image2
                    else:
                        # Positive examples



                        same_df=self.df[self.df['Class'] == selected_label]
                        image1,image2=self.get_hard_pair(df1=same_df,df2=same_df,same=True)

                        pairs[0][b, :, :, :] = image1
                        pairs[1][b, :, :, :] = image2

            else:

                for b in range(self.batch_size):
                    # Select anchor image
                    selected_label = np.random.choice(self.labels, 1)[0]

                    # Negative - 0 (Different images), Positive = 1 (Same images)
                    if targets[b] == 0:
                        # Negative examples
                        labels_ = self.labels.copy()
                        labels_.remove(selected_label)
                        target_label = np.random.choice(labels_, 1, replace=False)[0]

                        # load images into pairs
                        image1 = self.df[self.df["Class"] == selected_label].sample(n=1)['Path'].values[0]
                        image1 = self.load_image(image1)
                        image1 = self.histogram_equalization(image1)

                        image2 = self.df[self.df["Class"] == target_label].sample(n=1)['Path'].values[0]
                        image2 = self.load_image(image2)
                        image2 = self.histogram_equalization(image2)

                        pairs[0][b, :, :, :] = image1
                        pairs[1][b, :, :, :] = image2
                    else:
                        # Positive examples
                        images = self.df[self.df['Class'] == selected_label].sample(n=2)['Path'].values
                        image1 = self.load_image(images[0])
                        image1 = self.histogram_equalization(image1)

                        image2 = self.load_image(images[1])
                        image2 = self.histogram_equalization(image2)

                        pairs[0][b, :, :, :] = image1
                        pairs[1][b, :, :, :] = image2



            yield pairs, targets.astype(int)


batch_size = 6

train_gen = Data_Generator(df, image_shape, batch_size)
batch, targets = next(train_gen.get_batch())

mLabels = ["Different", "Same"]
for n in range(batch_size):
    show_images(random.sample([batch[0][n], batch[1][n]], 2), mLabels[targets[n]])

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.backend.tensorflow_backend import set_session
# import tensorflow.keras.models.model_from_json
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from sklearn.metrics import precision_recall_fscore_support

import warnings

warnings.filterwarnings('ignore')

base_model = load_model(join("weights", "vgg16_covid19_weights.hd5"))
base_model.summary()

classifer_model = new_model = tf.keras.models.Sequential(base_model.layers[:-8])

classifer_model.summary()


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def create_vgg_model(lr=1e-4):
    """
        Model architecture
    """
    input_shape = image_shape
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    pretrain_model = Model(inputs=classifer_model.input, outputs=classifer_model.layers[-1].output)
    for layer in pretrain_model.layers:  # Set all layers to be trainable
        layer.trainable = True
    for layer in pretrain_model.layers[-4:]:  # last 4 layer freeze
        layer.trainable = False

    model.add(pretrain_model)
    model.add(Flatten())
    model.add(Dense(5120, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(lr))
    print(siamese_net.summary())

    # return the model
    return siamese_net


model = create_vgg_model()
model._name = "SiameseNet_VGG16"

'''
import tensorflow.keras
import pydot as pyd
from IPython.display import SVG
from tensorflow.keras.utils.vis_utils import model_to_dot
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/dot.exe'

tensorflow.keras.utils.vis_utils.pydot = pyd

#create your model
#then call the function on your model
plot_model(model, to_file='SiameseNet_VGG16.png')
'''

batch_size = 30
weights_path = join("weights", "siamese_model.hd5")

checkpointer = ModelCheckpoint(weights_path, monitor="loss", verbose=1, mode='min', save_best_only=True)
reduceLROnPlato = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, verbose=1, mode='min')
earlyStop = EarlyStopping(monitor="loss", mode='min', verbose=1, patience=8)

generator = Data_Generator(df, image_shape, batch_size).get_batch()
validator = Data_Generator(df, image_shape, batch_size).get_batch()

steps_per_epoch = (len(df_X_train) // batch_size) * 2
validation_steps = len(df_X_test) // batch_size

print(steps_per_epoch)
print(validation_steps)

history = model.fit_generator(generator,
                              steps_per_epoch=(len(df_X_train) // batch_size) * 2,
                              validation_data=next(validator),
                              validation_steps=len(df_X_test) // batch_size,
                              epochs=90,
                              verbose=1,

                              callbacks=[checkpointer, reduceLROnPlato, earlyStop])