# Reference: http://solarisailab.com/archives/2647

import tensorflow as tf

# tf.keras.Model를 이용하여 CNN 모델을 정의
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        # First CL(Convolution Layer)
        # 5x5 Kernel Size with 32 Filters
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu")
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)