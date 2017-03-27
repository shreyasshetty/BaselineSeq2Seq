""" main.py : The main script for running the baseline models.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf

from BaseDataset import BaseDataset
from model import BaselineSeq2Seq
from model import evaluate_model
from model import generate_sentences
from BaseDataset import build_vocabulary

flags = tf.app.flags

flags.DEFINE_integer("vocab_size", 20000, "Vocabulary size")
flags.DEFINE_integer("embedding_size", 300, "Embedding size")
flags.DEFINE_integer("tokens_per_field", 5, "max tokens per field in an infobox")
flags.DEFINE_integer("fields_per_box", 10, "max fields in an infobox")
flags.DEFINE_integer("rnn_size", 128, "Size of the RNN hidden layer")
flags.DEFINE_integer("max_source_len", 50, "Size of the input infobox")
flags.DEFINE_integer("sum_seq_length", 30, "Max length of generated summary")
flags.DEFINE_integer("min_field_freq", 100, "Min freq of fields to be considered in vocab")
flags.DEFINE_float("learning_rate", 4e-4, "learning rate")
flags.DEFINE_string("optimizer", 'adam', "Optimizer to be used")
flags.DEFINE_integer("batch_size", 64, "Batch size of mini-batches")
flags.DEFINE_string("data_dir", "../data", "Path to the dataset directory")
flags.DEFINE_string("save_dir", "../experiment/", "Save the results in the following path")
flags.DEFINE_integer("num_epochs", 5, "number of epochs to run the experiment")
flags.DEFINE_integer("print_every", 100, "print the training loss every so many steps")
flags.DEFINE_integer("valid_every", 1000, "validate the model every so many steps")
flags.DEFINE_integer("test_every", 1000, "test the model every so many steps")
flags.DEFINE_integer("save_every_epochs", 1, "save the model every so many epochs")
flags.DEFINE_integer("gen_train_every", 1, "generate sentences on train dataset every so many epochs")
flags.DEFINE_integer("gen_test_every", 1, "generate sentences on test dataset every so many epochs")
flags.DEFINE_integer("gen_valid_every", 1, "generate sentences on valid dataset every so many epochs")
flags.DEFINE_integer("train_step_every", 1000, "generate sentences on train dataset every so many steps")
flags.DEFINE_integer("test_step_every", 1000, "generate sentences on test dataset every so many steps")
flags.DEFINE_integer("valid_step_every", 1000, "generate sentences on valid dataset every so many steps")
flags.DEFINE_integer("true_feed", 1, "change feed_previous to True for train dataset after these many epochs")

FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    with open(os.path.join(FLAGS.save_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(flags.FLAGS.__flags, indent=1))

    tr_infoboxes = os.path.join(FLAGS.data_dir, 'train', 'train.box')
    tr_sentences = os.path.join(FLAGS.data_dir, 'train', 'train_in.sent')

    te_infoboxes = os.path.join(FLAGS.data_dir, 'test', 'test.box')
    te_sentences = os.path.join(FLAGS.data_dir, 'test', 'test_in.sent')

    va_infoboxes = os.path.join(FLAGS.data_dir, 'valid', 'valid.box')
    va_sentences = os.path.join(FLAGS.data_dir, 'valid', 'valid_in.sent')

    # Checkpoint directory
    checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    batch_size = FLAGS.batch_size
    max_source_len = FLAGS.max_source_len
    sum_seq_len = FLAGS.sum_seq_length

    print("Building the word index")
    start = time.time()
    #word_to_id = build_vocabulary(tr_sentences,
    #                              FLAGS.vocab_size,
    #                              FLAGS.sum_seq_length)
    word_to_id = build_vocabulary(tr_infoboxes, tr_sentences, FLAGS.vocab_size,
                                  FLAGS.min_field_freq,
                                  FLAGS.fields_per_box,
                                  FLAGS.sum_seq_length)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    vocab_size = len(word_to_id)
    duration = time.time() - start
    print("Built index in %.3f s" %(duration))

    print("Building the training dataset object")
    start = time.time()
    train_dataset = BaseDataset(tr_infoboxes,
                                tr_sentences,
                                FLAGS.tokens_per_field,
                                max_source_len,
                                sum_seq_len,
                                word_to_id,
                                batch_size)
    duration = time.time() - start
    print("Built train dataset in %.5f s" %(duration))

    print("Building the test dataset object")
    start = time.time()
    test_dataset = BaseDataset(te_infoboxes,
                               te_sentences,
                               FLAGS.tokens_per_field,
                               max_source_len,
                               sum_seq_len,
                               word_to_id,
                               batch_size)
    duration = time.time() - start
    print("Built test dataset in %.5f s" %(duration))

    print("Building the valid dataset object")
    start = time.time()
    valid_dataset = BaseDataset(va_infoboxes,
                                va_sentences,
                                FLAGS.tokens_per_field,
                                max_source_len,
                                sum_seq_len,
                                word_to_id,
                                batch_size)
    duration = time.time() - start
    print("Built valid dataset in %.5f s" %(duration))

    with tf.Graph().as_default():
        tf.set_random_seed(1234)

        model = BaselineSeq2Seq(vocab_size,
                                FLAGS.embedding_size,
                                FLAGS.learning_rate,
                                FLAGS.optimizer,
                                FLAGS.rnn_size)

        enc_inputs = tf.placeholder(tf.int32,
                                    shape=(batch_size, max_source_len),
                                    name="encoder_inputs")
        dec_inputs = tf.placeholder(tf.int32,
                                    shape=(batch_size, sum_seq_len + 1),
                                    name="decoder_inputs")
        dec_weights = tf.placeholder(tf.float32,
                                     shape=(batch_size, sum_seq_len),
                                     name="decoder_weights")
        feed_previous = tf.placeholder(tf.bool, name="feed_previous")

        logits_op = model.inference_s2s_att(enc_inputs, dec_inputs,
                                            feed_previous)
        loss_op = model.loss(logits_op, dec_inputs, dec_weights)
        train_op = model.training(loss_op)

        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        init = tf.initialize_all_variables()
        sess.run(init)

        print "Trainable variables"
        print '\n'.join([v.name for v in tf.trainable_variables()])

        while train_dataset.epochs_done < FLAGS.num_epochs:
            start_e = time.time()
            for step in range(train_dataset.num_batches):
                benc_ins, bdec_ins, bdec_wts = train_dataset.next_batch()
                if train_dataset.epochs_done >= FLAGS.true_feed:
                    train_feed = True
                else:
                    train_feed = False
                feed_dict = {enc_inputs : benc_ins,
                             dec_inputs : bdec_ins,
                             dec_weights : bdec_wts,
                             feed_previous : train_feed}
                _, loss_val = sess.run([train_op, loss_op],
                                       feed_dict=feed_dict)
                perplexity = np.exp(float(loss_val)) if loss_val < 300 else float('inf')

                if step % FLAGS.print_every == 0:
                    with open(os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_train_loss.log'), 'a') as log_f:
                        log_f.write('epoch %d batch %d: loss = %.3f perplexity = %.2f\n' % (train_dataset.epochs_done + 1,
                                                                                            step,
                                                                                            loss_val,
                                                                                            perplexity))
                    print('epoch %d batch %d: loss = %.3f perplexity = %.2f ' % (train_dataset.epochs_done + 1,
                                                                                 step,
                                                                                 loss_val,
                                                                                 perplexity))

                if step % FLAGS.valid_every == 0 and step != 0:
                    v_loss, v_perp = evaluate_model(sess,
                                                    valid_dataset,
                                                    loss_op,
                                                    enc_inputs,
                                                    dec_inputs,
                                                    dec_weights,
                                                    feed_previous)
                    with open(os.path.join(FLAGS.save_dir, 'valid.log'), 'a') as log_f:
                        log_f.write('valid : epoch %d batch %d : loss = %0.3f perplexity = %0.3f\n' %(train_dataset.epochs_done + 1,
                                                                                                      step,
                                                                                                      v_loss,
                                                                                                      v_perp))
                    print('valid : epoch %d batch %d : loss = %0.3f perplexity = %0.3f\n' %(train_dataset.epochs_done + 1,
                                                                                            step,
                                                                                            v_loss,
                                                                                            v_perp))

                if step % FLAGS.test_every == 0 and step != 0:
                    t_loss, t_perp = evaluate_model(sess,
                                                    test_dataset,
                                                    loss_op,
                                                    enc_inputs,
                                                    dec_inputs,
                                                    dec_weights,
                                                    feed_previous)
                    with open(os.path.join(FLAGS.save_dir, 'test.log'), 'a') as log_f:
                        log_f.write('test : epoch %d batch %d : loss = %0.3f perplexity = %0.3f\n' %(train_dataset.epochs_done + 1,
                                                                                                     step,
                                                                                                     t_loss,
                                                                                                     t_perp))
                    print('test : epoch %d batch %d : loss = %0.3f perplexity = %0.3f\n' %(train_dataset.epochs_done + 1,
                                                                                           step,
                                                                                           t_loss,
                                                                                           t_perp))

                if step % FLAGS.train_step_every == 0 and step != 0:
                    epochs_done = train_dataset.epochs_done
					index_in_epoch = train_dataset.index_in_epoch
                    benc_ins, bdec_ins, bdec_wts, sents = train_dataset.next_batch_gen()
                    feed_dict = {enc_inputs : benc_ins,
                                 dec_inputs : bdec_ins,
                                 dec_weights : bdec_wts,
                                 feed_previous : True}
                    logits = np.array(sess.run(logits_op, feed_dict=feed_dict))
                    logits = np.reshape(logits, (FLAGS.batch_size, FLAGS.sum_seq_length, vocab_size))

                    save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_train.gen')
                    true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_train.true')

                    with open(save_path, 'a') as save_f:
                        for idx in xrange(batch_size):
                            words = []
                            for l in xrange(FLAGS.sum_seq_length):
                                tokenid = np.argmax(logits[idx, l])
                                words.append(id_to_word[tokenid])
                                save_f.write(' '.join(words) + '\n')
                    with open(true_path, 'a') as true_f:
                        for sent in sents:
                            true_f.write(sent)
                    train_dataset.reset_batch(index_in_epoch, epochs_done)

                if step % FLAGS.test_step_every == 0 and step != 0:
                    benc_ins, bdec_ins, bdec_wts, sents = test_dataset.next_batch_gen()
                    feed_dict = {enc_inputs : benc_ins,
                                 dec_inputs : bdec_ins,
                                 dec_weights : bdec_wts,
                                 feed_previous : True}
                    logits = np.array(sess.run(logits_op, feed_dict=feed_dict))
                    logits = np.reshape(logits, (FLAGS.batch_size, FLAGS.sum_seq_length, vocab_size))

                    save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_test.gen')
                    true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_test.true')

                    with open(save_path, 'a') as save_f:
                        for idx in xrange(batch_size):
                            words = []
                            for l in xrange(FLAGS.sum_seq_length):
                                tokenid = np.argmax(logits[idx, l])
                                words.append(id_to_word[tokenid])
                                save_f.write(' '.join(words) + '\n')
                    with open(true_path, 'a') as true_f:
                        for sent in sents:
                            true_f.write(sent)
                    test_dataset.reset_batch()

                if step % FLAGS.valid_step_every == 0 and step != 0:
                    benc_ins, bdec_ins, bdec_wts, sents = valid_dataset.next_batch_gen()
                    feed_dict = {enc_inputs : benc_ins,
                                 dec_inputs : bdec_ins,
                                 dec_weights : bdec_wts,
                                 feed_previous : True}
                    logits = np.array(sess.run(logits_op, feed_dict=feed_dict))
                    logits = np.reshape(logits, (FLAGS.batch_size, FLAGS.sum_seq_length, vocab_size))

                    save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_valid.gen')
                    true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done + 1) + '_' + str(step) + '_valid.true')

                    with open(save_path, 'a') as save_f:
                        for idx in xrange(batch_size):
                            words = []
                            for l in xrange(FLAGS.sum_seq_length):
                                tokenid = np.argmax(logits[idx, l])
                                words.append(id_to_word[tokenid])
                                save_f.write(' '.join(words) + '\n')
                    with open(true_path, 'a') as true_f:
                        for sent in sents:
                            true_f.write(sent)
                    valid_dataset.reset_batch()


            duration_e = time.time() - start_e
            with open(os.path.join(FLAGS.save_dir, 'time_taken.txt'), 'a') as time_f:
                time_f.write('Epoch : %d\tTime taken : %0.5f\n' %(train_dataset.epochs_done, duration_e))

            if train_dataset.epochs_done % FLAGS.save_every_epochs == 0:
                modelfile = os.path.join(checkpoint_dir, str(train_dataset.epochs_done) + '.ckpt')
                saver.save(sess, modelfile)

            if train_dataset.epochs_done % FLAGS.gen_valid_every == 0:
                true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_valid.true')
                save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_valid.gen')
                generate_sentences(sess, valid_dataset, logits_op, enc_inputs,
                                   dec_inputs, dec_weights, feed_previous,
                                   id_to_word, FLAGS.batch_size, FLAGS.sum_seq_length,
                                   vocab_size, save_path, true_path,
                                   only_num_batches=None)

            if train_dataset.epochs_done % FLAGS.gen_train_every == 0:
                true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_train.true')
                save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_train.gen')
                generate_sentences(sess, train_dataset, logits_op, enc_inputs,
                                   dec_inputs, dec_weights, feed_previous,
                                   id_to_word, FLAGS.batch_size, FLAGS.sum_seq_length,
                                   vocab_size, save_path, true_path,
                                   only_num_batches=10)

            if train_dataset.epochs_done % FLAGS.gen_test_every == 0:
                true_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_test.true')
                save_path = os.path.join(FLAGS.save_dir, str(train_dataset.epochs_done) + '_test.gen')
                generate_sentences(sess, test_dataset, logits_op, enc_inputs,
                                   dec_inputs, dec_weights, feed_previous,
                                   id_to_word, FLAGS.batch_size, FLAGS.sum_seq_length,
                                   vocab_size, save_path, true_path,
                                   only_num_batches=None)


if __name__ == "__main__":
    tf.app.run()
