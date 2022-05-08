from datetime import datetime

import numpy as np
import pywt
import umap
import tensorflow as tf
from PIL import Image
from abc import ABC, abstractmethod
from img2vec_pytorch import Img2Vec
from tensorflow import saved_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from pathlib import Path
import tensorflow_hub as hub

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features_from_generator(self, generator: Sequence) -> np.ndarray:
        """
        Extracts features from TF Generator
        Args:
            generator: TF Generator
        """
        pass

    @abstractmethod
    def extract_features_from_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Extracts features from a single batch of items
        Args:
            batch: nparray of items
        """
        pass

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Fits the extractor to given data
        Args:
            data: Data for fitting the extractor
        """
        pass
    @abstractmethod
    def extract_feature(self, data: np.ndarray) -> np.ndarray:
        """
        Fits the extractor to given data
        Args:
            data: Data for fitting the extractor
        """
        pass



class Img2VecExtractor(FeatureExtractor):
    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def __init__(self):
        """
        Initializes Img2Vec extractor
        """
        self.extractor = Img2Vec(cuda=True)

    def extract_features_from_batch(self, batch):
        batch_features = []

        for input in batch:
            npy_img = ((input + 1) * 127.5).astype('uint8')
            img = Image.fromarray(npy_img, 'RGB')
            vec = self.extractor.get_vec(img)
            batch_features.append(vec)

        return batch_features

    def extract_features_from_generator(self, generator):
        generator_features = []
        for batch, label in generator:
            batch_features = self.extract_features_from_batch(batch)
            generator_features.extend(batch_features)
        return generator_features

    def extract_feature(self, data: np.ndarray) -> np.ndarray:
        img = Image.fromarray(data, 'RGB')
        vec = self.extractor.get_vec(img)
        return vec

class Vgg16 (FeatureExtractor):
    def __init__(self):
        """
        Creates a feature extractor based on pretrained TF network
        Args:
            config: Config file used in main program
            conv_network: TF network defined in tensorflow.keras.applications
        """
        config_path = Path('src/anonymity/utils/config')
        model = load_model(config_path)
        model = Model(inputs=model.inputs, outputs=model.get_layer(index=2).output)
        self.model = model
        

    def extract_features_from_generator(self, generator):
        slide_features = self.model.predict_generator(generator=generator, verbose=1, max_queue_size=6, workers=8,
                                                      use_multiprocessing=True)
        return slide_features

    def extract_features_from_batch(self, batch):
        batch_features = self.model.predict_on_batch(batch)
        return batch_features

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        # data = (data-128)/128
        data = data/256
        result = self.model.predict(np.expand_dims(data, axis=0))
        return result

class SimCLR(FeatureExtractor):
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        hub_path = '/path/to/simclr/hub'

        self.model = hub.Module(hub_path, trainable=False)

    def extract_features_from_generator(self, generator):
        raise Exception("Not supported for this type of extractor")

    def extract_features_from_batch(self, batch):
        raise Exception("Not supported for this type of extractor")

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        data = data/255
        res = self.model(np.expand_dims(data, axis=0))
        
        return res

class PretrainedTensorflowNetwork(FeatureExtractor):
    def __init__(self, config: dict, preprocessing, conv_network=VGG16):
        """
        Creates a feature extractor based on pretrained TF network
        Args:
            config: Config file used in main program
            conv_network: TF network defined in tensorflow.keras.applications
        """
        
        self.config = config
        self.model = conv_network(weights='imagenet', include_top=False)

    def extract_features_from_generator(self, generator):
        slide_features = self.model.predict_generator(generator=generator, verbose=1, max_queue_size=6, workers=8,
                                                      use_multiprocessing=True)
        return slide_features

    def extract_features_from_batch(self, batch):
        batch_features = self.model.predict_on_batch(batch)
        return batch_features

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        
        return self.model.predict(preprocess_input_vgg16(np.expand_dims(data, axis=0)))


class Vgg16Pretrained(FeatureExtractor):
    def __init__(self, config):
        """
        Creates a feature extractor using network pretrained on classifying histopathological data
        Args:
            config: Config file used in main program
        """
        self.config = config
        pre_model = PretrainedModelAnonymityForEval(config, conv_net=VGG16)
        self.model = Model(inputs=pre_model.model.input, outputs=pre_model.model.layers[-2].output)

    def extract_features_from_generator(self, generator):
        slide_features = self.model.predict_generator(generator=generator, verbose=1, max_queue_size=6, workers=8,
                                                      use_multiprocessing=True)
        return slide_features

    def extract_features_from_batch(self, batch):
        batch_features = self.model.predict_on_batch(batch)
        return batch_features

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        return self.model.predict(np.expand_dims(data, axis=0))


class DWTFeatureExtractor(FeatureExtractor):
    def __init__(self, config: dict, wavelet: str = 'db1') -> None:
        """
        Creates a feature extractor that uses DWT
        Args:
            config: Config file used in main program
            wavelet: Wavelet to use
        """
        self.config = config
        self.wavelet = wavelet

    def extract_features_from_generator(self, generator):
        generator_features = []
        for batch, label in generator:
            batch_features = self.extract_features_from_batch(batch)
            for feature in batch_features:
                generator_features.append(feature)
            # feature = pywt.downcoef('d', input[0].flatten(), 'db1')
            # slide_features.append(feature)
        return generator_features

    def extract_features_from_batch(self, batch):
        batch_features = []
        for input in batch:
            rA, (rH, rV, rD) = pywt.dwt2(input[:, :, 0], self.wavelet)
            gA, (gH, gV, gD) = pywt.dwt2(input[:, :, 1], self.wavelet)
            bA, (bH, bV, bD) = pywt.dwt2(input[:, :, 2], self.wavelet)
            if self.config['model']['dwt']['direction'] == 'horizontal':
                feature = np.array([rH, gH, bH]).flatten()
            if self.config['model']['dwt']['direction'] == 'vertical':
                feature = np.array([rV, gV, bV]).flatten()
            if self.config['model']['dwt']['direction'] == 'diagonal':
                feature = np.array([rD, gD, bD]).flatten()
            if self.config['model']['dwt']['direction'] == 'all':
                feature = np.array([rH, gH, bH, rV, gV, bV, rD, gD, bD]).flatten()
            batch_features.append(feature)
        return np.array(batch_features)

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        raise Exception("Not supported for this type of extractor")


class UmapFeatureExtractor(FeatureExtractor):

    def fit_predict(self, data):
        return self.model.fit_transform(data)

    def __init__(self, dim: int = 50, neighbors: int = 15, min_dist: float = 0.1) -> None:
        """
        Creates an UMAP feature extractor
        Args:
            dim: Target dimension number
            neighbors: K-nearest neighbor hyperparameter
            min_dist: min_dist hyperparameter
        """
        self.model = umap.UMAP(n_neighbors=neighbors,
                               min_dist=min_dist,
                               n_components=dim
                               )

    def extract_features_from_generator(self, generator):
        generator_features = []
        for batch, label in generator:
            batch_features = self.extract_features_from_batch(batch)
            generator_features.extend(batch_features)
        return generator_features

    def extract_features_from_batch(self, batch):
        return self.model.transform(batch)

    def fit(self, data):
        self.model.fit(data)

    def extract_feature(self, data):
        raise Exception("Not supported for this type of extractor")


class AutoEncoder(FeatureExtractor):
    def __init__(self, config: dict = None, weights_path: str = None, feature_extractor: bool = False) -> None:
        """
        Creates a feature extractor with AutoEncoder
        Args:
            config: Config file used in main program
            weights_path: Path to a weight file if using pretrained weights
            feature_extractor: Prepare extracting output
        """
        self.create_cnn()
        if weights_path != None:
            self.autoencoder.load_weights(filepath=weights_path)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        self.autoencoder.build(input_shape=(512, 512, 3))
        self.autoencoder.summary()
        # keras.utils.plot_model(self.autoencoder, to_file='AE.png', show_shapes=True, dpi=1000)
        self.encoder = None
        if (feature_extractor):
            self.get_feature_extractor()
        if config is not None:
            config['model']['preprocess_function'] = lambda x: x / 255

    def create_cnn(self):
        """
        Creates an internal model for AutoEncoder
        """
        input_img = keras.Input(shape=(512, 512, 3))
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        # x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='encoding')(x)
        x = layers.UpSampling2D((2, 2))(x)
        # x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        # x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        self.autoencoder = keras.Model(input_img, decoded)

    def get_feature_extractor(self):
        """
        Creates a model for encoding features by accesing the encoding layer
        """
        new_output = self.autoencoder.get_layer('encoding')
        self.encoder = Model(inputs=self.autoencoder.input, outputs=new_output.output, )

    def train(self, generator: Sequence, epochs: int = 30, steps_per_epoch: int = 10000) -> None:
        """
        Trains an AutoEncoder on a given generator
        Args:
            generator: TF generator
            epochs: epoch number
            steps_per_epoch: steps for epoch
        """
        self.autoencoder.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
        self.autoencoder.save_weights(
            filepath=f"encoder_weights_temp-{datetime.now().strftime('%d-%b-%Y__%H:%M:%S')}.h5")

    def test(self, generator: Sequence) -> None:
        """
        Creates random 10 images using encoding and decoding input data, useful for testing if autoencoder is doing
        something meaningful.
        Args:
            generator: TF Generator
        """
        i = 0
        # generator has some random data, so running on first 10 images is OK for brief look if it does something
        for data, labels in generator:
            predicted = self.autoencoder.predict_on_batch(data)[0]

            npy_img = ((data[0]) * 255).astype('uint8')
            original_img = Image.fromarray(npy_img, 'RGB')
            original_img.save(
                f"samples/original-{i}-{datetime.now().strftime('%d-%b-%Y__%H:%M:%S')}.png")

            npy_img = ((predicted) * 255).astype('uint8')
            predicted_img = Image.fromarray(npy_img, 'RGB')
            predicted_img.save(
                f"samples/predicted-{i}-{datetime.now().strftime('%d-%b-%Y__%H:%M:%S')}.png")
            i += 1
            if i == 10:
                return

    def extract_features_from_generator(self, generator):
        slide_features = self.encoder.predict_generator(generator=generator, verbose=1, max_queue_size=6, workers=8,
                                                        use_multiprocessing=True)
        return slide_features

    def extract_features_from_batch(self, batch):
        batch_features = self.encoder.predict_on_batch(batch)
        return batch_features

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        return self.encoder.predict(np.expand_dims(data, axis=0))


class ResnetExtractor(FeatureExtractor):

    def fit_predict(self, data):
        raise Exception("Not supported for this type of extractor")

    def __init__(self, ) -> None:
        """
        Creates an Resnet50 feature extractor
        Args:
            dim: Target dimension number
            neighbors: K-nearest neighbor hyperparameter
            min_dist: min_dist hyperparameter
        """
        model = ResNet50(weights='imagenet', include_top=True)
        model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)
        self.model = model

    def extract_features_from_generator(self, generator):
        raise Exception("Not supported for this type of extractor")

    def extract_features_from_batch(self, batch):
        return self.model.transform(batch)

    def fit(self, data):
        raise Exception("Not supported for this type of extractor")

    def extract_feature(self, data):
        img = preprocess_input(np.expand_dims(data, axis=0))
        return self.model.predict(img)
