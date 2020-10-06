import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())

import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

content_layer = 'block5_conv2'



#style  layers in vgg 19 model
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def extract_from_model(model,image,x,y,filters,path,layer_name):
    m = model.predict(image)
    m1 = model.get_layer(layer_name).output
    img = np.zeros((x,y))
    p1 = tf.make_ndarray(m1)
    print(p1)
    for oo in range(filters):
        for i in range(x):
            for j in range(y):
                img[i][j] = p1[0][i][j][oo] 
        print(oo)

        cv2.imwrite("./" + path +"/layer_"+ str(oo)+".jpg",img)



# <<<<<------------------------------------------------------------------------model load ----------------------------------------------------------------------->>>>>>
model = tf.keras.applications.vgg19.VGG19()


#model.summary()
img1 = preprocess_image("content.jpg")
x = tf.keras.applications.vgg19.preprocess_input(
    img1, data_format=None
)


#<<<<<<<--------------------------------------------------------------------------extraction-------------------------------------------------------------------->>>>>>>>>

extract_from_model(model,img1,14,14,512,"block5_conv2",'block5_conv2')
extract_from_model(model,img1,224,224,64,"block1_conv1",'block1_conv2')

