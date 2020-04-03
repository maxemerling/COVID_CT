"""
Adapted from https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
"""

from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Dropout, concatenate
from keras.utils import get_file

WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

def fire_module(x, fire_id, squeeze=16, expand=64):
    fire_id = 'fire' + str(fire_id) + '/'
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=fire_id + 'squeeze1x1')(x)
    x = Activation('relu', name=fire_id + 'relu_squeeze1x1')(x)
    
    left = Convolution2D(expand, (1, 1), padding='valid', name=fire_id + 'expand1x1')(x)
    left = Activation('relu', name=fire_id + 'relu_expand1x1')(left)
 
    right = Convolution2D(expand, (3, 3), padding='same', name=fire_id + 'expand3x3')(x)
    right = Activation('relu', name=fire_id + 'relu_expand3x3')(right)
 
    x = concatenate([left, right], axis=3, name=fire_id + 'concat')
    
    return x

def SqueezeNet(input_shape, weights='imagenet'):
    inputs = Input(shape=input_shape)
    
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    model = Model(inputs, x, name='squeezenet')
    
    if weights == 'base_model':
        weights_path = 'base_model.h5'
    elif weights == 'imagenet':
        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        raise ValueError('Unknown `weights` argument')
    
    
    return model