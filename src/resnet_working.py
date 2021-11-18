from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class ResNet():
    @classmethod
    def learning_rate_schedule(cls, epoch):
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

    @classmethod
    def resnet_layer(cls, inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
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

    @classmethod
    def resnet_v2(cls, input_shape, depth, num_classes=10):
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = ResNet.resnet_layer(inputs=inputs,
                                num_filters=num_filters_in,
                                conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = ResNet.resnet_layer(inputs=x,
                                        num_filters=num_filters_in,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        conv_first=False)
                y = ResNet.resnet_layer(inputs=y,
                                        num_filters=num_filters_in,
                                        conv_first=False)
                y = ResNet.resnet_layer(inputs=y,
                                        num_filters=num_filters_out,
                                        kernel_size=1,
                                        conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = ResNet.resnet_layer(inputs=x,
                                            num_filters=num_filters_out,
                                            kernel_size=1,
                                            strides=strides,
                                            activation=None,
                                            batch_normalization=False)
                x = add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model
