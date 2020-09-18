"""
Module: Utilities
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""

import re
import json
import config
import tensorflow as tf
import tensorflow_datasets as tfds


class DataUtilities:
    """Getting the data, preprocess to remove / replace unsupported patterns and encode the data. """

    def __init__(self):
        """Importing the data from a json and initializing the variables used in the class. """
        with open("data.json") as jsonfile:
            self.data = json.load(jsonfile)
        self.qnatuples = self.data['Data']
        self.tokenized_questions = []
        self.tokenized_answers = []
        self.tokenizer = None
        self.dataset = None
        self.questions = []
        self.answers = []
        self.qns = []
        self.ans = []

    def preprocess(self, string):
        """Cleaning the data and removing unsupported patterns from the input. """
        string = string.lower()
        string = re.sub(r"([?.!,])", r" \1 ", string)

        return string

    def tokenize(self):
        """Tokenizing, encoding and padding to form the dataset for model training. """
        for (question, answer) in self.qnatuples:
            self.questions.append(self.preprocess(question))
            self.answers.append(self.preprocess(answer))
            if len(self.questions) >= config.hyperparams.max_convos:
                break
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=config.hyperparams.max_vocab_size)
        config.hyperparams.start_token = [self.tokenizer.vocab_size]
        config.hyperparams.stop_token = [self.tokenizer.vocab_size + 1]
        config.hyperparams.actual_vocab_size = self.tokenizer.vocab_size + 2

        for (q, a) in zip(self.questions, self.answers):
            self.qns = config.hyperparams.start_token + self.tokenizer.encode(q) + config.hyperparams.stop_token
            self.ans = config.hyperparams.start_token + self.tokenizer.encode(a) + config.hyperparams.stop_token
            if len(self.qns) <= config.hyperparams.max_length and len(self.ans) <= config.hyperparams.max_length:
                self.tokenized_questions.append(self.qns)
                self.tokenized_answers.append(self.ans)
        print(len(self.tokenized_questions))
        print(len(self.tokenized_answers))
        self.tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenized_questions, maxlen=config.hyperparams.max_length, padding='post')
        self.tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenized_answers, maxlen=config.hyperparams.max_length, padding='post')
        self.dataset = tf.data.Dataset.from_tensor_slices(({'XfIn': self.tokenized_questions,
                                                            'XfDeIn': self.tokenized_answers[:, :-1]},
                                                           self.tokenized_answers[:, 1:]))
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(len(self.tokenized_questions))
        self.dataset = self.dataset.batch(config.hyperparams.batch_size)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return self.dataset, self.tokenizer


class ScaledDotProductAttention:
    """Calculate the scaled dot product attention. The formula for calculating the scaled dot product attention is :
    ScaledDotProductAttention = softmax[(Q . K) / (D ** 0.5)] . V
    Mask is optional.
    """

    def __init__(self, q, k, v, mask):
        self.q = q
        self.k = k
        self.v = v
        self.mask = mask

    def attention(self):
        """Multiply Query and Key, scale it by square root of depth of key, softmax it and multiply by Value. """
        qk = tf.matmul(self.q, self.k, transpose_b=True)
        scale_factor = tf.cast(tf.shape(self.k)[-1], tf.float32)
        logits = qk / tf.math.sqrt(scale_factor)
        if self.mask is not None:
            logits += (self.mask * -1e9)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        sdpa = tf.matmul(attention_weights, self.v)
        return sdpa


class MultiHeadAttention(tf.keras.layers.Layer):
    """Split Query Key Value into multiple heads so that the transformer will be able to attend to information at
    different positions at different representational spaces. """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.depth = config.hyperparams.d_model // config.hyperparams.num_heads
        self.wq = tf.keras.layers.Dense(config.hyperparams.d_model)
        self.wk = tf.keras.layers.Dense(config.hyperparams.d_model)
        self.wv = tf.keras.layers.Dense(config.hyperparams.d_model)
        self.dense = tf.keras.layers.Dense(config.hyperparams.d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, config.hyperparams.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """Split query, key, value into multiple heads, calculate Scaled Dot Product Attention and concatenate those
        into a single attention. """
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = ScaledDotProductAttention(query, key, value, mask).attention()
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, config.hyperparams.d_model))
        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(tf.keras.layers.Layer):
    """Since the transformer is a non-recurrent model unlike RNNs and LSTMs, there has to be a positional encoding to
    denote the relative position of each word in the sequence. Otherwise, the transformer will effectively see a bag of
    wards with no information on the correlation of the words whatsoever. """

    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding()

    def get_angles(self, pos, i):
        """Defining the base formula for positional encoding. """
        angles = 1 / tf.pow(10000, (2 * (i // 2) / tf.cast(config.hyperparams.d_model, tf.float32)))
        return pos * angles

    def positional_encoding(self):
        """Positional Encoding for each word - sine to words at even indices and cosine to words at odd indices. """
        angle_rads = self.get_angles(tf.range(config.hyperparams.actual_vocab_size, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(config.hyperparams.d_model, dtype=tf.float32)[tf.newaxis, :])
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class Encoder:
    """Defining the encoder part of the transformer model. This will take in word embeddings and sums it up with the
    positional encoding. This aggregation is fed as an input to the encoder."""

    def __init__(self):
        """Initializing the variables used in the class."""
        self.en_inputs, self.el_inputs, self.en_outputs, self.el_outputs = None, None, None, None
        self.en_padding_mask, self.el_padding_mask = None, None
        self.en_embeddings, self.el_attention = None, None

    def encoder_layer(self):
        self.el_inputs = tf.keras.Input(shape=(None, config.hyperparams.d_model))
        self.el_padding_mask = tf.keras.Input(shape=(1, 1, None))
        self.el_attention = MultiHeadAttention()({'query': self.el_inputs,
                                                  'key': self.el_inputs,
                                                  'value': self.el_inputs,
                                                  'mask': self.el_padding_mask,
                                                  })
        self.el_attention = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.el_attention)
        self.el_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.el_inputs + self.el_attention)
        self.el_outputs = tf.keras.layers.Dense(config.hyperparams.num_units, activation='relu')(self.el_attention)
        self.el_outputs = tf.keras.layers.Dense(config.hyperparams.d_model)(self.el_outputs)
        self.el_outputs = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.el_outputs)
        self.el_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.el_attention + self.el_outputs)
        return tf.keras.Model(inputs=[self.el_inputs, self.el_padding_mask], outputs=self.el_outputs)

    def encoder(self):
        self.en_inputs = tf.keras.Input(shape=(None,))
        self.en_padding_mask = tf.keras.Input(shape=(1, 1, None))
        self.en_embeddings = tf.keras.layers.Embedding(
            config.hyperparams.actual_vocab_size, config.hyperparams.d_model)(self.en_inputs)
        self.en_embeddings *= tf.math.sqrt(tf.cast(config.hyperparams.d_model, tf.float32))
        self.en_embeddings = PositionalEncoding()(self.en_embeddings)
        self.en_outputs = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.en_embeddings)
        for i in range(config.hyperparams.num_layers):
            self.en_outputs = self.encoder_layer()([self.en_outputs, self.en_padding_mask])
        return tf.keras.Model(inputs=[self.en_inputs, self.en_padding_mask], outputs=self.en_outputs)


class Decoder:
    """Defining the decoder part of the transformer. The decoder takes in the output of the encoder, processes it,
    decodes it and displays it in a human readable format."""
    def __init__(self):
        """Initializing the variables used in the class."""
        self.de_padding_mask, self.dl_padding_mask, self.dl_en_outputs, self.de_en_outputs = None, None, None, None
        self.de_inputs, self.dl_inputs, self.de_outputs, self.dl_outputs = None, None, None, None
        self.de_embeddings, self.dl_attn1, self.dl_attn2 = None, None, None
        self.de_foresight_mask, self.dl_foresight_mask = None, None

    def decoder_layer(self):
        self.dl_inputs = tf.keras.Input(shape=(None, config.hyperparams.d_model))
        self.dl_en_outputs = tf.keras.Input(shape=(None, config.hyperparams.d_model))
        self.dl_foresight_mask = tf.keras.Input(shape=(1, None, None))
        self.dl_padding_mask = tf.keras.Input(shape=(1, 1, None))
        self.dl_attn1 = MultiHeadAttention()(inputs={'query': self.dl_inputs,
                                                     'key': self.dl_inputs,
                                                     'value': self.dl_inputs,
                                                     'mask': self.dl_foresight_mask
                                                     })
        self.dl_attn1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.dl_attn1 + self.dl_inputs)
        self.dl_attn2 = MultiHeadAttention()(inputs={'query': self.dl_attn1,
                                                     'key': self.dl_en_outputs,
                                                     'value': self.dl_en_outputs,
                                                     'mask': self.dl_padding_mask
                                                     })
        self.dl_attn2 = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.dl_attn2)
        self.dl_attn2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.dl_attn2 + self.dl_attn1)
        self.dl_outputs = tf.keras.layers.Dense(config.hyperparams.num_units, activation='relu')(self.dl_attn2)
        self.dl_outputs = tf.keras.layers.Dense(config.hyperparams.d_model)(self.dl_outputs)
        self.dl_outputs = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.dl_outputs)
        self.dl_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.dl_outputs + self.dl_attn2)
        return tf.keras.Model(inputs=[self.dl_inputs, self.dl_en_outputs, self.dl_foresight_mask, self.dl_padding_mask],
                              outputs=self.dl_outputs)

    def decoder(self):
        self.de_inputs = tf.keras.Input(shape=(None,))
        self.de_en_outputs = tf.keras.Input(shape=(None, config.hyperparams.d_model))
        self.de_foresight_mask = tf.keras.Input(shape=(1, None, None))
        self.de_padding_mask = tf.keras.Input(shape=(1, 1, None))
        self.de_embeddings = tf.keras.layers.Embedding(
            config.hyperparams.actual_vocab_size, config.hyperparams.d_model)(self.de_inputs)
        self.de_embeddings *= tf.math.sqrt(tf.cast(config.hyperparams.d_model, tf.float32))
        self.de_embeddings = PositionalEncoding()(self.de_embeddings)
        self.de_outputs = tf.keras.layers.Dropout(config.hyperparams.dropout)(self.de_embeddings)
        for i in range(config.hyperparams.num_layers):
            self.de_outputs = self.decoder_layer()(
                inputs=[self.de_outputs, self.de_en_outputs, self.de_foresight_mask, self.de_padding_mask])
        return tf.keras.Model(inputs=[self.de_inputs, self.de_en_outputs, self.de_foresight_mask, self.de_padding_mask],
                              outputs=self.de_outputs)


class Transformer:
    """Building the transformer from component parts. """

    def __init__(self):
        self.xf_inputs, self.xf_de_inputs, self.xf_en_outputs, self.xf_de_outputs = None, None, None, None
        self.xf_en_padding_mask, self.xf_foresight_mask, self.xf_de_padding_mask = None, None, None
        self.pad_mask, self.foresight_mask, self.seq_len, self.pad_mask2 = None, None, None, None
        self.xf_outputs = None

    def post_padding(self, x):
        self.pad_mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return self.pad_mask[:, tf.newaxis, tf.newaxis, :]

    def block_foresight(self, x):
        self.seq_len = tf.shape(x)[1]
        self.foresight_mask = 1 - tf.linalg.band_part(tf.ones((self.seq_len, self.seq_len)), -1, 0)
        self.pad_mask2 = self.post_padding(x)
        return tf.maximum(self.foresight_mask, self.pad_mask2)

    def transformer(self):
        self.xf_inputs = tf.keras.Input(shape=(None,), name='XfIn')
        self.xf_de_inputs = tf.keras.Input(shape=(None,), name='XfDeIn')
        self.xf_en_padding_mask = tf.keras.layers.Lambda(
                            self.post_padding, output_shape=(1, 1, None), name='XfEnPM')(self.xf_inputs)
        self.xf_foresight_mask = tf.keras.layers.Lambda(
                            self.block_foresight, output_shape=(1, None, None), name='XfFm')(self.xf_de_inputs)
        self.xf_de_padding_mask = tf.keras.layers.Lambda(
                            self.post_padding, output_shape=(1, 1, None), name='XfDePm')(self.xf_inputs)
        self.xf_en_outputs = Encoder().encoder()(inputs=[self.xf_inputs, self.xf_en_padding_mask])
        self.xf_de_outputs = Decoder().decoder()(
            inputs=[self.xf_de_inputs, self.xf_en_outputs, self.xf_foresight_mask, self.xf_de_padding_mask])
        self.xf_outputs = tf.keras.layers.Dense(units=config.hyperparams.actual_vocab_size)(self.xf_de_outputs)
        return tf.keras.Model(inputs=[self.xf_inputs, self.xf_de_inputs], outputs=self.xf_outputs)
