import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.losses import BinaryCrossentropy as bcLoss
from keras.optimizers import Adam

# Load the Model 
model = keras.models.load_model('./model.h5')
model.trainable = False
optimizer = Adam(learning_rate=0.001)
loss_object = bcLoss()

# Start with a blank image, and then modify it step by step in order to
# obtain a valid ID
path = './blank.png'


def process_image(path):
    image = tf.keras.utils.load_img(
        path,
        target_size=(96, 128),
        color_mode='grayscale',
        interpolation='bilinear',
    )
    # plt.imshow(image)
    # plt.show()
    processedImage = tf.keras.utils.img_to_array(image)
    processedImage = np.array([processedImage]) / 255
    return processedImage


# This function returns the gradient of the loss w.r.t the input image
# The gradient will be used to learn "the other way" in order to
# create an adversarial pattern
def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        input_image = tf.convert_to_tensor(input_image)
        input_label = tf.convert_to_tensor([input_label])
        tape.watch(input_image)
        prediction = model(input_image, training=False)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image
    gradient = tape.gradient(loss, input_image)

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# Given a label, the function returns the image modified in order to increase
# the loss. This means that the model will be more likely to predict the
# opposite label
def modify_image(image, label, eps):
    perturbations = create_adversarial_pattern(image, label)
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    return adv_x


# Given an image, the function returns the prediction of the model
def predict(processedImage):
    p = model.predict(processedImage)[0][0]
    print("{:.20f}".format(p))
    status = round(p)
    return status


def save_image(processedImage, filename='data.png'):
    data = np.asarray([processedImage])
    data = np.reshape(data, (96, 128, 1))
    data = data * 255
    tf.keras.utils.save_img(filename, data, scale=False)


img = process_image(path)
eps = 0.05

for step in range(0, 10):
    img = modify_image(img, 1, eps)
    modelPred = predict(img)
    print(modelPred)


save_image(img, filename='fake_signature.png')

status = predict(img)
if status == 0:
    print("Access Granted")
else:
    print("Access Denied")