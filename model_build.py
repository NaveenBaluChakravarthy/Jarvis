"""
Module: Model Preparation 01
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""

import config
import tensorflow as tf
from utilities import DataUtilities, Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(config.hyperparams.d_model, dtype=tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (config.hyperparams.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, config.hyperparams.max_length - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, config.hyperparams.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


if __name__ == '__main__':
    """Build the model and save as h5 file. """

    tf.random.set_seed(config.hyperparams.tf_seed)
    data_util = DataUtilities()
    dataset, tokenizer = data_util.tokenize()

    jarvis = Transformer().transformer()
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    jarvis.compile(optimizer, loss=loss_function, metrics=[accuracy])
    jarvis.fit(dataset, epochs=config.hyperparams.epochs)

    txt = 'who is the father of tony'
    txt = tf.expand_dims(config.hyperparams.start_token + tokenizer.encode(txt) + config.hyperparams.stop_token, axis=0)
    outs = tf.expand_dims(config.hyperparams.start_token, 0)
    predictions = jarvis(txt, outs)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    outs = tf.concat([outs, predicted_id], axis=-1)
    output = tf.squeeze(outs, axis=0)
    ansz = tokenizer.decode([i for i in output if i < config.hyperparams.actual_vocab_size])
    print(ansz)
