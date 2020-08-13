import numpy as np

index = np.load('./dataset/cifar/index.npy')

Speckle = np.load('./dataset/cifar/cifar10_speckles_128.npy')
print(Speckle.shape)
Speckle = Speckle[index]
X_train = Speckle[0:8000,:,:,:]
X_train = 2 * (X_train/255) - 1
print(X_train.shape)
np.save("./dataset/cifar/split/speckle_train.npy", X_train)
#
X_val = Speckle[8000:9000,:,:,:]
X_val = 2 * (X_val/255) - 1
print(X_val.shape)
np.save("./dataset/cifar/split/speckle_validation.npy", X_val)

X_test = Speckle[9000:10000,:,:,:]
X_test = 2 * (X_test/255) - 1
print(X_test.shape)
np.save("./dataset/cifar/split/speckle_test.npy", X_test)
#
#
Image = np.load('./dataset/cifar/cifar10_32.npy')
print(Image.shape)
Image = Image[index]
Y_train = Image[0:8000,:,:,:]
Y_train = 2 * (Y_train/255) - 1
print(Y_train.shape)
np.save("./dataset/cifar/split/image_train.npy", Y_train)

Y_val = Image[8000:9000,:,:,:]
Y_val = 2 * (Y_val/255) - 1
print(Y_val.shape)
np.save("./dataset/cifar/split/image_validation.npy", Y_val)

Y_test = Image[9000:10000,:,:,:]
Y_test = 2 * (Y_test/255) - 1
print(Y_test.shape)
np.save("./dataset/cifar/split/image_test.npy", Y_test)
