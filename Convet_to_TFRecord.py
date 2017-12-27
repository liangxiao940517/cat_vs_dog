import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd='F:/Python/dataset_dog_cat\\'
#cwd='F:/Python/test\\'
t_cwd='F:/Python/test_dataset_dog_cat\\'
twd="F:/Python/kaggle_dog_cat/test1\\"
classes={'cat','dog'}
train_filename="cat_dog_train.tfrecords"
test_filename="cat_dog_test.tfrecords"
predict_filename="cat_dog_predict.tfrecords"
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

with tf.python_io.TFRecordWriter(train_filename) as writer:
    for index,name in enumerate(classes):
        class_path=cwd+name+'\\'
        for img_name in os.listdir(class_path):
            img_path=class_path+img_name
            img=Image.open(img_path)
            img=img.resize((256,256))
            img_raw=img.tobytes()
            example=tf.train.Example(features=tf.train.Features(feature={
                    "label":_int64_feature(index),
                    "img_raw":_bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())
            
            
with tf.python_io.TFRecordWriter(test_filename) as writer:
    for index,name in enumerate(classes):
        class_path=t_cwd+name+'\\'
        for img_name in os.listdir(class_path):
            img_path=class_path+img_name
            img=Image.open(img_path)
            img=img.resize((256,256))
            img_raw=img.tobytes()
            example=tf.train.Example(features=tf.train.Features(feature={
                    "label":_int64_feature(index),
                    "img_raw":_bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())


with tf.python_io.TFRecordWriter(predict_filename) as writer:
    for img_name in os.listdir(twd):
        img_path=twd+img_name
        img=Image.open(img_path)
        img=img.resize((256,256))
        img_raw=img.tobytes()
        example=tf.train.Example(features=tf.train.Features(feature={
#                'label':_int64_feature(0),
                'img_raw':_bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())       
