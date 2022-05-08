from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from feature_extractors import FeatureExtractor, ResnetExtractor, Img2VecExtractor, AutoEncoder, PretrainedTensorflowNetwork, SimCLR
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

def get_extractor(type):
    if type == 'resnet':
        extractor = ResnetExtractor()
    elif type == 'img2vec':
        extractor = Img2VecExtractor()
    elif type == 'vgg16':
        config = {'model': {
            'input_shape': (224,224,3)
        }}
        extractor = PretrainedTensorflowNetwork(config, nasnet_preprocess)
    elif type == 'inception':
        config = {'model': {
            'input_shape': (224,224,3)
        }}
        extractor = PretrainedTensorflowNetwork(config, inception_preprocess, conv_network=InceptionV3)
    elif type == 'clr':
        extractor = SimCLR()
    else:
        raise Exception("Not supported type of extractor: {}".format(type))
    return extractor
