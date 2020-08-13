from tensorflow.keras.models import save_model, load_model, Model
from tensorflow.keras.layers import Input, BatchNormalization, UpSampling2D, Reshape, LeakyReLU, Add
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Concatenate, Dropout



def encoder(code_length=128):
    inputs = Input(shape=(128, 128, 1))
    conv0 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='lecun_normal')(inputs)
    conv0 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv1)
    conv2 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    dense5 = Flatten()(conv4)
    dense5 = Dense(code_length)(dense5)
    outputs = BatchNormalization()(dense5)

    model = Model(inputs=inputs, outputs=outputs, name='encoder')
    return model


def decoder(code_length=128):
    inputs = Input(shape=(code_length,))
    dense1 = Dense(256 * 8 * 8)(inputs)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    conv1 = Reshape([8, 8, 256])(dense1)
    conv1 = Conv2D(128, 2, padding='same', kernel_initializer='lecun_normal')(UpSampling2D(size=(2, 2))(conv1))
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(128, 3, padding='same', kernel_initializer='lecun_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(64, 2, padding='same', kernel_initializer='lecun_normal')(UpSampling2D(size=(2, 2))(conv1))
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    outputs = Conv2D(1, 1, activation='tanh', padding='same')(conv3)

    model = Model(inputs=inputs, outputs=outputs, name='decoder')
    return model


def encoder_decoder(enc, dec):
    inputs = Input(shape=(128, 128, 1))
    outputs = dec(enc(inputs))
    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet():
    inputs = Input(shape=(128, 128, 1))
    conv0 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='lecun_normal')(inputs)
    conv0 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv1)
    conv2 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', strides=2, padding='same', kernel_initializer='lecun_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='lecun_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = Concatenate(axis=-1)([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='lecun_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=-1)([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='lecun_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=-1)([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='lecun_normal')(conv7)
    outputs = Conv2D(1, 1, activation='tanh', padding='same', kernel_initializer='lecun_normal')(conv8)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def mlp(input_length=1024, hidden_units=1024, hidden_layers=2, activation="relu", batchnorm=True, dropout=0):
    inputs = Input(shape=(input_length,))
    x = inputs
    features = []
    for i in range(hidden_layers):
        x = Dense(hidden_units)(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        if dropout:
            x = Dropout(dropout)(x)

        feature = Reshape([32, 32, 1])(x)
        features.append(feature)

    outputs = Concatenate(axis=-1)(features)

    # outputs = Dense(1024)(x)
    # outputs = Reshape([32, 32, 1])(x)
    outputs = Conv2D(16, 5, activation='relu', padding='same')(outputs)
    outputs = Conv2D(16, 3, activation='relu', padding='same')(outputs)
    outputs = Conv2D(1, 1, padding='same')(outputs)

    outputs = Activation('tanh')(outputs)
    model = Model(inputs, outputs)
    return model

# if __name__ == '__main__':
#     model = encoder()
#     model.save("encoder.h5")
#     model = decoder()
#     model.save("decoder.h5")
#     model = unet()
#     model.save("unet.h5")