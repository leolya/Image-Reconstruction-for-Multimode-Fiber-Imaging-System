import numpy as np
from skimage import measure
from tensorflow.keras.callbacks import Callback


def assess(model, X_test, Y_test, flatten=False):
    test_num = X_test.shape[0]
    ssim_sum = 0
    mse_sum = 0
    psnr_sum = 0
    predict = model.predict(X_test)
    if flatten:
        predict = np.reshape(predict, (test_num, 32, 32, 1))
        Y_test = np.reshape(Y_test, (test_num, 32, 32, 1))

    for j in range(test_num):
        # original image
        img1 = np.squeeze(Y_test[j])
        img1 = 255*(img1 + 1)/2
        # reconstruction
        img2 = np.squeeze(predict[j])
        img2 = 255*(img2 + 1)/2
        # compute SSIM
        ssim = measure.compare_ssim(img1,img2,data_range=255)
        ssim_sum = ssim_sum + ssim
        # compute PSNR
        psnr = measure.compare_psnr(img1,img2,data_range=255)
        psnr_sum = psnr_sum + psnr
        # compute MSE
        mse = measure.compare_mse(img1,img2)
        mse_sum = mse_sum + mse

    ssim_average = ssim_sum / test_num
    mse_average = mse_sum / test_num
    psnr_average = psnr_sum / test_num
    print('\nSSIM：', ssim_average, " MSE: ", mse_average, " PSNR: ", psnr_average)
    return ssim_average, mse_average, psnr_average


class Test(Callback):
    def __init__(self, X_val, Y_val, check_point, flatten):
        Callback.__init__(self)
        self.X_val = X_val
        self.Y_val = Y_val
        self.ssim_max = 0
        self.check_point = check_point
        self.flatten = flatten

    def on_epoch_end(self, epoch, logs=None):
        ssim, mse, psnr = assess(self.model, self.X_val, self.Y_val, self.flatten)
        if ssim >= self.ssim_max:
            self.ssim_max = ssim
            self.model.save(self.check_point)
        print("\ncurrent best ssim: ", self.ssim_max)

