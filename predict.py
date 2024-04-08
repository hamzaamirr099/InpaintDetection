
from __future__ import print_function
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import feature
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def FPNRes101e200(image_shape, class_number):
    resnet101Backbone = get_backbone_ResNet101(input_shape=image_shape)
    model = customFeaturePyramid2(resnet101Backbone, class_number)
    return model

"""## Data preprocessing """

IMG_SIZE = 256
    
def local_binary_pattern(image):
    radius = 2
    n_points = 8 * radius
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_lbp = feature.local_binary_pattern(img_gray, n_points,radius, method="uniform")
    
    return img_lbp


def laplacian(src, ddepth = cv2.CV_16S, kernel_size = 3):
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # [convert_to_gray]
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    # [convert]
    return abs_dst


def add_channels(image, lbp, laplace):
    lbp = np.array(lbp).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    merged = cv2.merge([gray,laplace,lbp])
    return merged


model_list = [FPNRes101e200] #FPNRes50e100


def get_backbone_ResNet101(input_shape):
    """Builds ResNet101 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet101(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


class customFeaturePyramid2(keras.models.Model):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50, ResNet101 and V1 counterparts.
    """

    def __init__(self, backbone=None, class_number=2, **kwargs):
        super(customFeaturePyramid2, self).__init__(name="customFeaturePyramid2", **kwargs)
        self.backbone = backbone if backbone else get_backbone_ResNet101()
        self.class_number = class_number
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.dense_d1 = keras.layers.Dense(64,
                                           activation='relu',
                                           kernel_initializer='he_uniform')
        self.dense_d2 = keras.layers.Dense(self.class_number,
                                           activation='sigmoid',
                                           kernel_initializer='he_normal')

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        p3_output = keras.layers.Flatten()(p3_output)
        p4_output = keras.layers.Flatten()(p4_output)
        p5_output = keras.layers.Flatten()(p5_output)
        p6_output = keras.layers.Flatten()(p6_output)
        p7_output = keras.layers.Flatten()(p7_output)
        m1_output = keras.layers.Concatenate(axis=1)([p3_output,
                                                      p4_output,
                                                      p5_output,
                                                      p6_output,
                                                      p7_output])
        m1_output = keras.layers.Flatten()(m1_output)
        m1_output = self.dense_d1(m1_output)
        m1_output = self.dense_d2(m1_output)
        return m1_output

opt = Adam()
loss = 'binary_crossentropy'
num_classes = 1
img_shape = (256, 256, 3)

model = FPNRes101e200(img_shape, num_classes)

def predictImage(imagePath):
    
    image = cv2.imread(imagePath)
    # Extracting features
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lbp = local_binary_pattern(image)
    laplace = laplacian(image)
    # mix = lbp * laplace
    image = add_channels(image, lbp, laplace)
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


    model.load_weights('Weights/NScifarFPNRes101e200_epoch5_batch10_weights')
    out1 = model.predict(image)
    nsScore = out1[0][0]
    
    # else:
    model.load_weights('Weights/SMcifarFPNRes101e200_epoch5_batch10_weights')
    out2 = model.predict(image)
    smScore = out2[0][0]
    
    print("shiftmap:")
    print(smScore)
    print("Navier:")
    print(nsScore)

    if nsScore > smScore:
        return nsScore
    else:
        return smScore

