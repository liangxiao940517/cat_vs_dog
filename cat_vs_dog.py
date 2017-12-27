import tensorflow as tf

filenames_test = "cat_dog_test.tfrecords"
filenames_train = "cat_dog_train.tfrecords"
filenames_prediction = "cat_dog_predict.tfrecords"

learning_rate = 0.0001
num_classes = 2
dropout1 = 0.5
dropout2 = 0.3


def parse(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'img_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                    })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256*256*3])
    image = tf.cast(image, tf.float32)/255-0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

def train_input_fn():
    dataset = tf.data.TFRecordDataset([filenames_train])
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(20000)
    dataset = dataset.repeat(8)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

def test_input_fn():
    dataset = tf.data.TFRecordDataset([filenames_test])
    dataset = dataset.map(parse)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

def conv_net(features, n_classes, dropout, reuse, is_training):
    x = tf.reshape(features, shape=[-1, 256, 256, 3])
    
    conv1 = tf.layers.conv2d(x, 96, 11, strides = (4,4), activation = tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 3, 2)
    
    conv2 = tf.layers.conv2d(conv1, 256, 5, activation = tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 3, 2)
    
    conv3 = tf.layers.conv2d(conv2, 384, 3, activation = tf.nn.relu)
    
    conv4 = tf.layers.conv2d(conv3, 384, 3, activation = tf.nn.relu)
    
    conv5 = tf.layers.conv2d(conv4, 256, 3, activation = tf.nn.relu)
    
    fc1 = tf.contrib.layers.flatten(conv5)
    
    fc1 = tf.layers.dense(fc1, 2048)
    fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
    
    fc1 = tf.layers.dense(fc1,1024)
    fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
    
    out = tf.layers.dense(fc1, n_classes)
    return out
    
def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout1, reuse = False, is_training = True)
    logits_test = conv_net(features, num_classes, dropout1, reuse = True, is_training = False)
    
    pred_classes = tf.argmax(logits_test, axis = 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)
        
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits_train, labels = tf.cast(labels, dtype = tf.int32)))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())
    
    acc_op = tf.metrics.accuracy(labels = labels, predictions = pred_classes)
    
    estim_specs = tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = pred_classes,
            loss = loss_op,
            train_op = train_op,
            eval_metric_ops = {'accuracy': acc_op})
    return estim_specs

model = tf.estimator.Estimator(model_fn)

model.train(train_input_fn, steps =160000)

evaluate = model.evaluate(test_input_fn)
print("Testing Accuracy:", evaluate['accuracy'])
