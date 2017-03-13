""" BaseDataset.py : Implements the data parser for
    the baseline model.
"""
import numpy as np
from vocabulary import field_name


def parse_box(box, tokens_per_field, max_source_len):
    """ parse_box : Read an input box and generate a list
    of tokens in the infobox in the order of appearance in the
    infobox.

    Args:
        box              : The input infobox
        tokens_per_field : The maximum number of tokens to be considered
                           for a given field.
        max_source_len   : maximum length of input infobox (represented as
                           a list)

    Returns:
        source           : A list of tokens appearing in the infobox.
                           If number of tokens is less than max_source_len
                           we extend the input by adding <PAD> symbols.
    """
    tokens = box.split()
    fields = []
    f2t = {}
    source = []

    for token in tokens:
        field, value = token.split(':', 1)
        if value != '<none>' and value != ',' and value != '.':
            fname = field_name(field)
            if fname in fields:
                f2t[fname].append(value)
            else:
                fields.append(fname)
                f2t[fname] = [value]

    for field in fields:
        toks = f2t[field]
        if len(toks) < tokens_per_field:
            source.extend(toks)
        else:
            source.extend(toks[:tokens_per_field])

    if len(source) > max_source_len:
        source = source[:max_source_len]
    else:
        source.extend(['<PAD>'] * (max_source_len - len(source)))

    assert len(source) == max_source_len

    return source


class BaseDataset(object):
    """ BaseDataset : Defines the dataset object for the baseline model.

    Functions implemented :
        next_batch()   - Returns the next batch for training
        reset_batch()  - Resets the dataset. Equivalent to starting the first
                         epoch
        num_batches    - Returns number of batches (based on batch size)
        num_examples   - Returns the number of examples in the dataset
        epochs_done    - Returns number of epochs done
    """
    def __init__(self, info_path, sent_path,
                 tokens_per_field,
                 max_source_len,
                 max_sum_len,
                 word_to_id,
                 batch_size):
        with open(info_path) as info_f:
            self.boxes = info_f.readlines()

        with open(sent_path) as sent_f:
            self.sentences = sent_f.readlines()

        self.input_box = []
        self.decoder_inputs = []
        self.decoder_weights = []
        dec_in = []
        dec_wts = []

        for (box, sent) in zip(self.boxes, self.sentences):
            source = parse_box(box, tokens_per_field, max_source_len)
            box_in = [word_to_id.get(word, 2) for word in source]

            swords = sent.split()
            swords = ['<GO>'] + swords
            num_valid = len(swords)
            dec_in = [word_to_id.get(word, 2) for word in swords]
            dec_wts = []

            if len(dec_in) >= max_sum_len + 1:
                dec_in = dec_in[:max_sum_len]
                dec_in.extend([word_to_id.get('<EOS>')])
            else:
                dec_in.extend([word_to_id.get('<EOS>')])
                dec_in.extend([word_to_id.get('<PAD>')] *
                              (max_sum_len + 1 - len(dec_in)))

            assert len(dec_in) == max_sum_len + 1

            dec_wts = [1] * num_valid
            if num_valid < max_sum_len:
                dec_wts.extend([0] * (max_sum_len - num_valid))
            else:
                dec_wts = dec_wts[:max_sum_len]

            self.input_box.append(box_in)
            self.decoder_inputs.append(dec_in)
            self.decoder_weights.append(dec_wts)

        # floor of number of batches
        num = len(self.sentences) // batch_size

        if len(self.sentences) % batch_size == 0:
            num_batches = num
        else:
            num_batches = num + 1
            # Extend the inputs to fit in the batch
            num_ex = len(self.input_box)
            total = num_batches * batch_size
            extend = total - num_ex
            in_last = self.input_box[-1]
            dec_in_last = self.decoder_inputs[-1]
            dec_wt_last = self.decoder_weights[-1]

            for _ in range(extend):
                self.input_box.append(in_last)
                self.decoder_inputs.append(dec_in_last)
                self.decoder_weights.append(dec_wt_last)

        self._num_batches = num_batches

        # Consistency check
        assert len(self.input_box) == self._num_batches * batch_size
        assert len(self.decoder_inputs) == self._num_batches * batch_size
        assert len(self.decoder_weights) == self._num_batches * batch_size

        self.input_box = np.asarray(self.input_box, dtype=np.int32)
        self.decoder_inputs = np.asarray(self.decoder_inputs,
                                         dtype=np.int32)
        self.decoder_weights = np.asarray(self.decoder_weights,
                                          dtype=np.float32)

        self.batch_size = batch_size
        self._num_examples = len(self.sentences)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._permutation = np.random.permutation(self._num_batches)

    def next_batch(self):
        """ next_batch : Generate the next batch for the inference procedure.
        """
        batch_size = self.batch_size
        start = self._permutation[self._index_in_epoch]
        end = start + batch_size
        self._index_in_epoch += 1

        if self._index_in_epoch == self._num_batches - 1:
            self._index_in_epoch = 0
            self._epochs_completed += 1
            self._permutation = np.random.permutation(self._num_batches)

        enc_inputs = self.input_box[start:end]
        dec_inputs = self.decoder_inputs[start:end]
        dec_weights = self.decoder_weights[start:end]

        return enc_inputs, dec_inputs, dec_weights

    def reset_batch(self):
        """ reset_batch : Reset the dataset. Equivalent to the
        condition at the start of the 1 epoch.
        """
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._permutation = np.random.permutation(self._num_batches)

    @property
    def num_examples(self):
        """ num_examples : Return the number of examples in the dataset.
        """
        return len(self.sentences)

    @property
    def num_batches(self):
        """ num_batches : Return the number of batches in the dataset.
        """
        return self._num_batches

    @property
    def epochs_done(self):
        return self._epochs_completed
