from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import numpy as np


class ResNet():
    def __init__(self, input_shape, depth, num_classes=10):
        self.input_shape = input_shape
        self.depth = depth
        self.num_classes = num_classes
        self.model = None
        self._build()

    def learning_rate_schedule(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def _build(self):
        if (self.depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        num_filters_in = 16
        num_res_blocks = int((self.depth - 2) / 9)
        inputs = Input(shape=self.input_shape)
        x = self._resnet_layer(
            inputs=inputs, num_filters=num_filters_in, conv_first=True)
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:
                        strides = 2
                y = self._resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides,
                                       activation=activation, batch_normalization=batch_normalization, conv_first=False)
                y = self._resnet_layer(
                    inputs=y, num_filters=num_filters_in, conv_first=False)
                y = self._resnet_layer(
                    inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
                if res_block == 0:
                    x = self._resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1,
                                           strides=strides, activation=None, batch_normalization=False)
                x = add([x, y])
            num_filters_in = num_filters_out
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(self.num_classes, activation='softmax',
                        kernel_initializer='he_normal')(y)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _resnet_layer(self, inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                      padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate_schedule(0)), metrics=['accuracy'])

    def train(self, x_train, y_train, batch_size, num_epochs, val_data=None, shuffle=True):
        lr_scheduler = LearningRateScheduler(self.learning_rate_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
            0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks = [lr_reducer, lr_scheduler]
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                       validation_data=val_data, shuffle=shuffle, callbacks=callbacks)

    def summary(self):
        self.model.summary()
