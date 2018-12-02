import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Lambda
import pickle
import numpy as np
from LRN import LRN
import cv2

def build_cnn(weights):

    model = Sequential()

    model.add(ZeroPadding2D(1, input_shape=(32,32, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(Conv2D(128, kernel_size=(8, 8)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(LRN(alpha=256,k=0,beta=0.5,n=256))

    model.set_weights(weights)

    return model


def build_L2_net(net_name):

    python_net_data = pickle.load(open("python_net_data/" + net_name + ".p", "rb"))
    return build_cnn(python_net_data['weights']), build_cnn(python_net_data['weights_cen']), python_net_data['pix_mean'], python_net_data['pix_mean_cen']


def cal_L2Net_des(net_name, testPatchs, flagCS = False):

    """
    Get descriptors for one or more patches

    Parameters
    ----------
    net_name : string
        One of "L2Net-HP", "L2Net-HP+", "L2Net-LIB", "L2Net-LIB+", "L2Net-ND", "L2Net-ND+", "L2Net-YOS", "L2Net-YOS+",
    testPatchs : array
        A numpy array of image data with deimensions (?, 32, 32, 1) or if flagCS (?, 64, 64, 1)
    flagCS : boolean
        Use a concated network one for the whole the patch and one for the center of the patch

    Returns
    -------
    descriptor
        Numpy array with size (?, 128) or if flagCS (?, 256)

    """

    model, model_cen, pix_mean, pix_mean_cen = build_L2_net(net_name)

    # print(model.summary())
    # print(model_cen.summary())

    if flagCS:

        testPatchsCen = testPatchs[:,16:48,16:48,:]
        testPatchsCen = testPatchsCen - pix_mean_cen
        testPatchsCen = np.array([(testPatchsCen[i] - np.mean(testPatchsCen[i]))/(np.std(testPatchsCen[i]) + 1e-12) for i in range(0, testPatchsCen.shape[0])])

        testPatchs = np.array([cv2.resize(testPatchs[i], (32,32), interpolation = cv2.INTER_CUBIC) for i in range(0, testPatchs.shape[0])])
        testPatchs = np.expand_dims(testPatchs, axis=-1)

        print(testPatchs)

    testPatchs = testPatchs - pix_mean
    testPatchs = np.array([(testPatchs[i] - np.mean(testPatchs[i]))/(np.std(testPatchs[i]) + 1e-12) for i in range(0, testPatchs.shape[0])])
    
    print(testPatchs.shape)

    res = np.reshape(model.predict(testPatchs), (testPatchs.shape[0], 128))

    if flagCS:
        
        resCen = np.reshape(model_cen.predict(testPatchsCen), (testPatchs.shape[0], 128))

        return np.concatenate((res, resCen), 1)

    else:

        return res 


data = np.full((1,64,64,1), 0.)

result = cal_L2Net_des("L2Net-HP+", data, flagCS=True)

print(result)