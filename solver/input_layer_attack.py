import keras
import numpy as np
from PIL import Image
import tensorflow as tf

from keras.layers import Input, Dense, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam


# Load the target Model and make it untrainable 
target_model = keras.models.load_model('./model.h5')
target_model.trainable = False

# Create the fake signature generator network. It takes as input the same
# vector that the target network would ouput (in our case, one boolean value
# for access granted or denied)
attack_input = Input(shape=(1,))
attack_model = Sequential()

# Now create a layer that will generate a 96x128x1 image (flattended and reshaped)
attack_model = Dense(96 * 128 * 1, activation='relu', input_dim=1)(attack_input)
attack_img = Reshape((96, 128, 1))(attack_model)
attack_model = Model(attack_input, attack_img)

# Combine both models. Attack Network | Target Network
# Note that the output of the attack model is the input of the target model
# And viceversa
target_output = target_model(attack_img)
combined_model = Model(attack_input, target_output)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.01, 0.5))

combined_model.summary()


batch_size = 32
total_epochs = 1000

# Now create the target "access granted" vector. This vector is the same for
# the input of the attack model and the output of the target model
# Because the target model is untrainable, our attack model will learn the pattern
# that the target model will use to determine if the input is a valid signature
access_granted_vector = np.zeros((batch_size, 1))
for i in range(batch_size):
    access_granted_vector[i][0] = 0.1

for x in range(total_epochs):
    combined_model.train_on_batch(access_granted_vector, access_granted_vector)
    if x % (int(total_epochs / 10)) == 0:
        print('Epoch ' + str(x) + ' / ' + str(total_epochs))


attack_model.save('./attack_model.h5')
# attack_model = keras.models.load_model('./attack_model.h5')


# Now using the trained attack model, generate the fake signature
fake_signature = attack_model.predict(access_granted_vector)
fake_signature = np.asarray([fake_signature[0]])
fake_signature = np.reshape(fake_signature, (96, 128, 1))
fake_signature = fake_signature * 255
image = tf.keras.utils.save_img('./fake_signature.png', fake_signature, scale=False)
