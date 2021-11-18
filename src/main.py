from six.moves import cPickle
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from resnet_working import ResNet
from resnet import ResNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

ROWS = 10


BATCH_SIZE = 32
EPOCHS = 100
USE_AUGMENTATION = True
NUM_CLASSES = 10
COLORS = 3
SUBTRACT_PIXEL_MEAN = True
DEPTH = COLORS * 9 + 2


if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = x_train.astype("uint8")

    fig, axes1 = plt.subplots(ROWS, ROWS, figsize=(10, 10))
    for j in range(ROWS):
        for k in range(ROWS):
            i = np.random.choice(range(len(x)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(x[i:i+1][0])

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if SUBTRACT_PIXEL_MEAN:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # Create the neural network
    model = ResNet.resnet_v2(input_shape=input_shape, depth=DEPTH)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=ResNet.learning_rate_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
