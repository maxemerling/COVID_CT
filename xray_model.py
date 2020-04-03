from keras_squeezenet import SqueezeNet

from keras.layers import Dropout, Convolution2D, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf

from collections import Counter

WIDTH, HEIGHT = (664, 485)

DROPOUT=0.2
CLASSES=2
BATCH_SIZE=16
NUM_EPOCHS=20
INIT_LR=0.0001

BASE_PATH = 'data/chest_xray/'
TRAIN_PATH = BASE_PATH + 'train'
VAL_PATH = BASE_PATH + 'val'
TEST_PATH = BASE_PATH + 'test'

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE)

counter = Counter(train_generator.classes)                        
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

base_model = SqueezeNet(input_shape=(HEIGHT, WIDTH, 3), weights="base_model")
x = base_model.output

x = Dropout(DROPOUT, name='drop9')(x)
x = Convolution2D(CLASSES, (1, 1), padding='valid', name='conv10')(x)
x = Activation('relu', name='relu_conv10')(x)
x = GlobalAveragePooling2D()(x)
predictions = Activation('softmax', name='loss')(x)

# x = GlobalAveragePooling2D()(x)
# x = Dropout(DROPOUT, name='drop9')(x)
# predictions = Dense(CLASSES, activation='softmax')(x)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer=Adam(lr=INIT_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights,
    shuffle=True
)

results = model.evaluate_generator(generator=test_generator,
                         steps=test_generator.samples // BATCH_SIZE)
print('RESULTS:', {key: val for key, val in zip(model.metrics_names, results)})
    
model.save('model.h5')
