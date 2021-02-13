"""
Module: Model Preparation 01
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""


import config
import random
random.seed(123)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
from utilities import DataUtilities, Transformer


if __name__ == '__main__':
    """Build the model and save as h5 file. """

    jarvis_callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)]
    data_util = DataUtilities()
    dataset, tokenizer = data_util.tokenize()
    jarvis = Transformer().transformer()
    jarvis.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    jarvis.fit(dataset, epochs=config.hyperparams.epochs, verbose=0, callbacks=jarvis_callback, shuffle=True,
               use_multiprocessing=False)
    jarvis_accuracy = jarvis.history.history.get('sparse_categorical_accuracy')[-1]
    if jarvis_accuracy >= 0.5:
        jarvis.save('jarvis_model.h5')
    print(f"Accuracy : {jarvis_accuracy}")

    txt = 'who is the father of tony'
    txt = tf.expand_dims(config.hyperparams.start_token + tokenizer.encode(txt) + config.hyperparams.stop_token, axis=0)
    outs = tf.expand_dims(config.hyperparams.start_token, axis=0)
    predictions = jarvis([txt, outs])
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    outs = tf.concat([outs, predicted_id], axis=-1)
    output = tf.squeeze(outs, axis=0)
    ansz = tokenizer.decode([i for i in output if i < tokenizer.vocab_size])
    print(ansz)
