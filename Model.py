import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from tensorflow import keras
import os


class Model:
    SIZE = 224  # size of images

    def __init__(self):
        self._model = None
        self._train_batches = None

    def resize_image(self, img, label):
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (self.SIZE, self.SIZE))
        img = img / 255.0
        return img, label

    def create_model(self, base_layer):
        # downloading training dataset
        train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
        train_resized = train[0].map(self.resize_image)
        self._train_batches = train_resized.shuffle(1000).batch(16)

        if base_layer:
            base_layers = tf.keras.applications.MobileNetV2(input_shape=(self.SIZE, self.SIZE, 3), include_top=False)
            base_layers.trainable = False
            self._model = tf.keras.Sequential([
                base_layers,
                GlobalAveragePooling2D(),
                Dropout(0.2),
                Dense(1)
            ])
            self._model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                metrics=['accuracy'])
        else:
            self._model = tf.keras.Sequential([
                Dropout(0.2),
                Dense(1)
            ])
            self._model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                                metrics=['accuracy'])

    def train(self, epochs=1, base_layer=True):
        self.create_model(base_layer)
        self._model.fit(self._train_batches, epochs=epochs)

    def save_trained(self, name):
        f_name = f'{name}.h5'
        path = os.path.join('saved_models', f_name)
        self._model.save(path)
        print(f'Successfully saved model into file {path}')

    def load_trained(self, path=r'trained_models\model1.h5'):
        """

        :param path: path to trained model folder or .h5 file; default value is model with 1 epoch of training
        :return: None
        """
        self._model = keras.models.load_model(path)

    def predict_many(self, path):
        """

        :param path: path to dir with images to predict; dir must contain only images!
        :return: array with (prediction, file_path)
        """
        res = []
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                img_path = os.path.join(path, file)
                img = load_img(img_path)
                img_array = img_to_array(img)
                _ = None
                img_resized, _ = self.resize_image(img_array, _)
                img_expended = np.expand_dims(img_resized, axis=0)
                prediction = self._model.predict(img_expended)[0][0]
                result = 'КОТ' if prediction < 0.5 else 'СОБАКА'
                res.append((result, img_path))

        return res

    def predict_one(self, path):
        """

        :param path: path to image to predict
        :return: string with result of prediction
        """
        img = load_img(path)
        img_array = img_to_array(img)
        _ = None
        img_resized, _ = self.resize_image(img_array, _)
        img_expended = np.expand_dims(img_resized, axis=0)
        prediction = self._model.predict(img_expended)[0][0]

        return ('КОТ' if prediction < 0.5 else 'СОБАКА'), path

    def test(self):
        dir_path_cats = r'test_images\cats'
        dir_path_dogs = r'test_images\dogs'
        mistakes = 0
        m = Model()
        # cats
        for file in os.listdir(dir_path_cats):
            if os.path.isfile(os.path.join(dir_path_cats, file)):
                img_path = os.path.join(dir_path_cats, file)
                img = load_img(img_path)
                img_array = img_to_array(img)
                _ = None
                img_resized, _ = self.resize_image(img_array, _)
                img_expended = np.expand_dims(img_resized, axis=0)
                prediction = self._model.predict(img_expended)[0][0]
                pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'

                if pred_label == 'СОБАКА':
                    mistakes += 1

        # dogs
        for file in os.listdir(dir_path_dogs):
            if os.path.isfile(os.path.join(dir_path_dogs, file)):
                img_path = os.path.join(dir_path_dogs, file)
                img = load_img(img_path)
                img_array = img_to_array(img)
                _ = None
                img_resized, _ = self.resize_image(img_array, _)
                img_expended = np.expand_dims(img_resized, axis=0)
                prediction = self._model.predict(img_expended)[0][0]
                pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'

                if pred_label == 'КОТ':
                    mistakes += 1

        return (2000 - mistakes) * 100 / 2000  # accuracy


if __name__ == '__main__':
    pass
