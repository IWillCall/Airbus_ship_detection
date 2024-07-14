#!/usr/bin/env python
# coding: utf-8

"""
Airbus Ship Detection Challenge - Model Training Script

This script trains a U-Net model for ship detection in satellite images.
It includes data preprocessing, model definition, and training.

Main components:
1. Data loading and preprocessing
2. Custom data generator for batch processing
3. U-Net model architecture
4. Training loop with callbacks
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy\

BATCH_SIZE = 64
IMAGE_SIZE = (256, 256)

def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decode a Run-Length Encoded (RLE) binary mask into a 2D numpy array.

    Parameters:
    - mask_rle (str): The RLE-encoded string representing the binary mask.
    - shape (tuple, optional): The shape of the target 2D array. Default is (768, 768).

    Returns:
    - numpy.ndarray: A 2D binary array representing the decoded mask.
    """
    if mask_rle == 'nan':
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T # Needed to align to RLE direction

train_df = pd.read_csv('data/train_ship_segmentations_v2.csv')
train_images_path = 'data/train_v2/'

# All data preprocessing is encapsulated in one transformer
class ShipDataTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer class for preprocessing ship data.
    
    This class adds new features to the dataset, such as the number of ships
    per image and the file size of each image.
    """
    def __init__(self, train_images_path):
        self.train_images_path = train_images_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['HasShip'] = X['EncodedPixels'].notna().apply(lambda x: 1 if x == True else 0)
        
        df_grouped = X.groupby('ImageId').agg({
            'EncodedPixels': lambda x: ','.join(x.astype('str')),
            'HasShip': 'sum'
        }).reset_index()
        
        df_grouped.rename(columns={'HasShip': 'ShipsAmount'}, inplace=True)
        
        df_grouped['FileSizeKb'] = df_grouped['ImageId'].map(
            lambda c_img_id: os.stat(os.path.join(self.train_images_path, c_img_id)).st_size/1024
        )
        
        df_grouped = df_grouped[df_grouped['FileSizeKb'] > 50]
        
        return df_grouped

ship_data_pipeline = Pipeline([
    ('ship_data_transformer', ShipDataTransformer(train_images_path))
])

train_df_transformed = ship_data_pipeline.fit_transform(train_df)


# Splitting data into training and test datasets. Here we also undersample images without ships
def split_data(data, test_size=0.2, empty_masks_percent=0.1, seed=42):
    """
    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    - seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 1.

    Returns: The training and testing sets.
    """

    all_masks_df = data.copy()

    empty_masks_df = all_masks_df[all_masks_df['ShipsAmount'] == 0]
    ship_masks_df = all_masks_df[all_masks_df['ShipsAmount'] > 0]
    
    empty_masks_amount = int(ship_masks_df.shape[0] * empty_masks_percent)
    print(f'Number of empty masks  - {empty_masks_amount}')
    
    all_masks_df = pd.concat([ship_masks_df, empty_masks_df.sample(n=empty_masks_amount, random_state=seed)], axis=0)

    # We stratify the data split to ensure that each dataset maintains the distribution of the original
    train_ids, test_ids = train_test_split(all_masks_df, test_size=test_size, stratify=all_masks_df['ShipsAmount'].values,
                                           random_state=seed)

    train_data = data[data['ImageId'].isin(train_ids.ImageId)]
    test_data = data[data['ImageId'].isin(test_ids.ImageId)]

    return train_data, test_data


train_data, val_data = split_data(train_df_transformed, test_size=0.2)

print(f'Number of masks in train data - {train_data.shape[0]}')
print(f'Number of masks in test data - {val_data.shape[0]}')


# Data generator for feeding batches to the model during training
class CustomDataGenerator(Sequence):
    def __init__(self, img_folder, csv_file, batch_size=32, img_size=(768, 768)):
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.data = csv_file

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        X = []
        y = []

        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.img_folder, row['ImageId'])

            # Load image
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0  # Normalize

            # Decode RLE to mask
            all_masks = self.data[self.data['ImageId'] == row['ImageId']].values[0][1].split(',')
            mask = np.zeros(self.img_size)
            for m in all_masks:
                decoded_mask = rle_decode(m)
                mask += cv2.resize(decoded_mask, self.img_size)

            mask = mask.astype(float)

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)

train_generator = CustomDataGenerator(img_folder=train_images_path,
                                      csv_file=train_data,
                                      batch_size=BATCH_SIZE,
                                      img_size=IMAGE_SIZE)

val_generator = CustomDataGenerator(img_folder=train_images_path,
                                    csv_file=val_data,
                                    batch_size=BATCH_SIZEE,
                                    img_size=IMAGE_SIZE)

# Building the U-Net model
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


model = build_unet()

model.summary()


# Callbacks that adjust the learning rate in case the model stalls
def create_callbacks():
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.33, 
        patience=1, 
        verbose=1, 
        mode='min', 
        min_delta=0.0001, 
        cooldown=0, 
        min_lr=1e-8
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        verbose=1, 
        patience=10
    )
    return [reduce_lr, early_stopping]

    
# Definition of the Dice coefficient metric function
def dice_coefficient(y_true, y_pred, smooth=1e-5):
    """
    Calculate the Dice coefficient for model evaluation.

    Parameters:
    - y_true (tensor): Ground truth masks.
    - y_pred (tensor): Predicted masks.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - float: The Dice coefficient.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if y_true.shape.ndims == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if y_pred.shape.ndims == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.reduce_sum(tf.square(y_true), axis=[1,2,3]) + tf.reduce_sum(tf.square(y_pred), axis=[1,2,3])
    return (2. * intersection + smooth) / (union + smooth)


# Model compilation
def compile_model(model, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss=binary_crossentropy, 
        metrics=[dice_coefficient]
    )
    return model


# Model training
def train_model(model, train_generator, val_generator, epochs=20):
    callbacks = create_callbacks()
    steps_per_epoch = len(train_generator)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    return history


model = build_unet()
model = compile_model(model)
history = train_model(model, train_generator, val_generator)