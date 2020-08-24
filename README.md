# Image Reconstruction for Multimode Fiber Imaging System

Implementation of three deep learning methods to inverse the distortion caused by Multimode Fiber, including U-net, autoencoder and MLP.

![](D:\github\Image-Reconstruction-for-Multimode-Fiber-Imaging-System\assets\problem.jpg)



### Training:

Specify the paths of training set and testing set and run main.py.

Dataset is separately stored in .npy files.

(sent email to leo.liyuang@gmail.com to ask for the dataset)

| File                   | shape               |
| ---------------------- | ------------------- |
| speckle_train.npy      | (3500, 128, 128, 1) |
| speckle_validation.npy | (500, 128, 128,1)   |
| speckle_test.npy       | (1000, 128, 128, 1) |
| image_train.npy        | (3500, 32, 32, 1)   |
| image_validation.npy   | (500, 32, 32, 1)    |
| image_test.npy         | (1000, 32, 32, 1)   |



### Results:

|      | U-net  | autoencoder | MLP    |
| ---- | ------ | ----------- | ------ |
| SSIM | 0.5865 | 0.6513      | 0.6742 |



### Visualization:

![](D:\github\Image-Reconstruction-for-Multimode-Fiber-Imaging-System\assets\visualization.jpg)

