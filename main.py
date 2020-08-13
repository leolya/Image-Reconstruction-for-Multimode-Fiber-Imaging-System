from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.optimizers import Adam
from model import encoder, decoder, encoder_decoder, unet, mlp
from utils import Test, assess
from sklearn.decomposition import PCA

speckle_path = ["./dataset/fashion mnist/speckle_train.npy",
                "./dataset/fashion mnist/speckle_validation.npy",
                "./dataset/fashion mnist/speckle_test.npy"]
image_path = ["./dataset/fashion mnist/image_train.npy",
              "./dataset/fashion mnist/image_validation.npy",
              "./dataset/fashion mnist/image_test.npy"]
check_point = "./models/check_point.h5"
decoder_path = "/content/drive/My Drive/Colab Notebooks/fashionMnist/autoencoder_master/decoder/decoder_128.h5"
save_path = "./models/"

def train(speckle_path, image_path, check_point, decoder_path, save_path, code_length=128, lr=1e-3,
          batch_size=64, epochs=100, use_pretrain=True, model_type='autoencoder'):
    X_train = np.load(speckle_path[0])
    Y_train = np.load(image_path[0])
    X_val = np.load(speckle_path[1])
    Y_val = np.load(image_path[1])

    if model_type == 'autoencoder':
        E = encoder(code_length=code_length)

        if use_pretrain:
            D = load_model(decoder_path)
        else:
            D = decoder(code_length=code_length)

        model = encoder_decoder(E, D)

    elif model_type == 'unet':
        model = unet()

    elif model_type == 'mlp':
        model = mlp(input_length=1024, hidden_units=1024, hidden_layers=7, activation="relu", batchnorm=True, dropout=0.1)

        X_train = np.reshape(X_train, (3500, 16384))
        # Y_train = np.reshape(Y_train, (3500, 1024))
        X_val = np.reshape(X_val, (500, 16384))
        # Y_val = np.reshape(Y_val, (500, 1024))

        pca = PCA(n_components=1024)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

        flatten = False


    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                      callbacks=[Test(X_val, Y_val, check_point, flatten)], shuffle=True)

    np.save(save_path + 'loss.npy', history.history['loss'])
    np.save(save_path + 'val_loss.npy', history.history['val_loss'])

    X_test = np.load(speckle_path[2])
    Y_test = np.load(image_path[2])

    if model_type == 'mlp':
        X_test = np.reshape(X_test, (1000, 16384))
        # Y_test = np.reshape(Y_test, (1000, 1024))
        X_test = pca.transform(X_test)


    model = load_model(check_point)
    print("final result:")
    a,b,c = assess(model, X_test, Y_test, flatten)


if __name__ == '__main__':
    train(speckle_path=speckle_path, image_path=image_path, check_point=check_point, decoder_path=decoder_path, save_path=save_path,
          code_length=128, lr=1e-3, batch_size=64, epochs=100, use_pretrain=False, model_type='mlp')









