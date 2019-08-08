from cifar import Cifar
from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import model
import numpy as np
import tensorflow as tf
import helper

def plot_embedding(X, y, title=None, imgName=None):

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    print(y)    
    print(type(y))
    print(len(y))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color='C{}'.format(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    # If title is not given, we assign training_mode to the title.
    plt.title(title)
    plt.savefig(imgName)

n_classes = 10
data_name = 'cifar10'
batch_size = 32
image_size = 32

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# ====================
# config model
# ====================
print('Load pretrained model...')

# extracted features
extractor = tf.layers.flatten(model.conv5)

# classifier
#weights = tf.Variable(tf.zeros([9216, n_classes]), name="output_weight")
#bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
#model = tf.matmul(extractor, weights) + bias
#outputs = tf.placeholder(tf.float32, [None, n_classes])

# ====================
# config dataset
# ====================
print('Prepare dataset')
train_dataset = Cifar(batch_size = batch_size)

init = tf.initialize_all_variables()

trian_features = None
train_label = None
with tf.Session(config=config) as sess:
    sess.run(init)
    for i in tqdm(range(20), unit=" batch "):
        this_batch = train_dataset.batch(i)
        train_X, train_y = helper.reshape_batch(this_batch, (image_size, image_size), n_classes)
        train_y = [np.argmax(element) for element in train_y]
        features = sess.run(
            [extractor],
            feed_dict={
                model.input_images: train_X 
            })
        if trian_features is None:
            trian_features = features[0]
            train_label = train_y
        else:
            trian_features = np.concatenate((trian_features, features[0]), axis=0)
            train_label += train_y


train_embedded = TSNE(n_components=2).fit_transform(trian_features)
plot_embedding(train_embedded, train_label, imgName='Training_Feature.jpg')