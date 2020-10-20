# Reference: http://solarisailab.com/archives/2647

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
x_train, x_test = x_train / 255., x_test / 255.

y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(10000).batch(50)
train_data_iter = iter(train_data)

# tf.keras.Model를 이용하여 CNN 모델을 정의
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        # First CL(Convolution Layer)
        # 5x5 Kernel size with 32 filters
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu")
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        # Second CL
        # 5x5 Kernel size with 64 filters
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation="relu")
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        # FL(Fully Connected Layer, or FCL)
        # Convert 64 activation maps with 7x7 size into 1024 features
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(1024, activation="relu")

        # Output Layer
        # Convert 1024 features to 0~9 with one-hot encoding
        self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x):
        # MNIST 데이터를 3차원으로 reshape
        # MNIST 데이터는 grayscale 이미지이기 때문에, 컬러채널의 값은 1
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # 28x28x1 -> 28x28x32
        h_conv1 = self.conv_layer_1(x_image)

        # 28x28x32 -> 14x14x32
        h_pool1 = self.pool_layer_1(h_conv1)

        # 14x14x32 -> 14x14x64
        h_conv2 = self.conv_layer_2(h_pool1)

        # 14x14x64 -> 7x7x64
        h_pool2 = self.pool_layer_2(h_conv2)

        # 7x7x64 -> 1024
        h_pool2_flat = self.flatten_layer(h_pool2)
        h_fc1 = self.fc_layer_1(h_pool2_flat)

        # 1024 -> 10
        logits = self.output_layer(h_fc1)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits

@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.Adam(1e-4)

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x)
        loss = cross_entropy_loss(logits, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

CNN_model = CNN()
epoch = 5000

for i in range(epoch):
    batch_x, batch_y = next(train_data_iter)

    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_model(batch_x)[0], batch_y)
        print("Epoch: %d, Accuracy: %f" % (i, train_accuracy))

    train_step(CNN_model, batch_x, batch_y)

print("Accuracy: %f" % compute_accuracy(CNN_model(x_test)[0], y_test))