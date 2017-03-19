""" model.py :
    Define the baseline models
    for biography generation.
"""

import numpy as np
import tensorflow as tf
from seq2seq import embedding_attention_seq2seq


class BaselineSeq2Seq(object):
    """ BaselineSeq2Seq : Specify the basic
    seq2seq model for biography generation.
    This model does not include attention mechanism.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 learning_rate,
                 optimizer,
                 rnn_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.rnn_size = rnn_size
        self.cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        self.projection_B = tf.get_variable(name="proj_b", shape=[vocab_size]) 
        self.projection_W = tf.get_variable(name="proj_w", shape=[rnn_size, vocab_size])

    def inference_s2s(self, encoder_inputs, decoder_inputs, feed_previous):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        cell = self.cell
        embedding_tied_rnn_seq2seq = tf.nn.seq2seq.embedding_tied_rnn_seq2seq

        encoder_inputs = tf.unpack(encoder_inputs, axis=1)
        decoder_inputs = tf.unpack(decoder_inputs, axis=1)

        outputs, _ = embedding_tied_rnn_seq2seq(encoder_inputs,
                                                decoder_inputs,
                                                cell,
                                                vocab_size,
                                                embedding_size,
                                                feed_previous=feed_previous)
        return outputs[:-1]

    def inference_s2s_att(self, encoder_inputs, decoder_inputs, feed_previous):
        vocab_size = self.vocab_size
        embedding_size = self.embedding_size
        cell = self.cell

        encoder_inputs = tf.unpack(encoder_inputs, axis=1)
        decoder_inputs = tf.unpack(decoder_inputs, axis=1)
        output_projection=(self.projection_W, self.projection_B)

        outputs, _ = embedding_attention_seq2seq(encoder_inputs,
                                                 decoder_inputs,
                                                 cell,
                                                 vocab_size,
                                                 vocab_size,
                                                 embedding_size,
                                                 feed_previous=feed_previous,
												 output_projection=output_projection)
        outputs = outputs[:-1]
        fin_outputs = [tf.matmul(o, self.projection_W) + self.projection_B for o in outputs]
        return fin_outputs

    def loss(self, logits, decoder_inputs, target_weights):
        """ Setup the loss op
        """
        decoder_inputs = tf.unpack(decoder_inputs, axis=1)
        targets = [decoder_inputs[i+1]
                   for i in xrange(len(decoder_inputs) - 1)]
        target_weights = tf.unpack(target_weights, axis=1)

        return tf.nn.seq2seq.sequence_loss(logits, targets, target_weights)

    def training(self, loss):
        """ Set up the training ops
        """
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = self.learning_rate
        optimizer = self.optimizer

        if optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'momentum':
            # Hard coded momentum parameter
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5)
        else:
            raise Exception('Optimizer not supported: {}'.format(optimizer))

        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        return train_op

    def prediction(self, logits):
        """ Setup the prediction op
        """
        logits = tf.pack(logits, axis=1)
        logit_shape = tf.shape(logits)
        logits = tf.reshape(logits, [-1, self.vocab_size])
        predictions = tf.nn.softmax(logits)
        return tf.reshape(predictions, logit_shape)


def evaluate_model(sess, dataset, loss_op, enc_inputs,
                   dec_inputs, dec_weights, feed_previous):
    num_batches = dataset.num_batches
    batch_loss = 0.0
    for i in xrange(num_batches):
        benc_ins, bdec_ins, bdec_wts = dataset.next_batch()
        feed_dict = { enc_inputs : benc_ins,
                      dec_inputs : bdec_ins,
                      dec_weights : bdec_wts,
                      feed_previous : True
                    }
        batch_loss += sess.run(loss_op, feed_dict=feed_dict)
    dataset.reset_batch()
    loss = batch_loss / num_batches
    perplexity = np.exp(float(loss)) if loss < 300 else float('inf')
    return loss, perplexity
