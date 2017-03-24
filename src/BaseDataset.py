""" BaseDataset.py : Implements the data parser for
    the baseline model.
"""
# pylint: disable=no-member
import numpy as np
from collections import Counter


def field_name(field):
    """ Extract the field name from the
    field token as extracted from the tokenized
    infobox.

    Eg: field_name("birthplace_1") == "birthplace"
        field_name("birthplace") == "birthplace"
        field_name("first_name_1") == "first_name"
    """
    # Dataset specific split.
    # split the field name based on '_' token
    ftokens = field.split("_")
    if ftokens[-1].isdigit():
        # The last entry captures the token position
        ftokens = ftokens[:-1]
    name = '_'.join(ftokens)
    return name


def build_vocabulary(info_path, sent_path, top_k, min_field_freq,
                     fields_per_box, sum_seq_len):
    """ Build a word_to_id index.
    This will be a common index with words coming from top_k
    words and fields coming from min_field_freq.
    """
    with open(info_path) as info_f:
        boxes = info_f.readlines()

    field_count = Counter()

    for box in boxes:
        tokens = box.split()
        fields = []
        for tok in tokens:
            field, value = tok.split(':', 1)
            if value != '<none>' and value != ',' and value != '.':
                fname = field_name(field)
                if fname not in fields:
                    fields.append(fname)
        if len(fields) > fields_per_box:
            fields = fields[:fields_per_box]
        field_count.update(fields)

    fname_count = field_count.most_common()
    flist = [fn for (fn, count) in fname_count if count >= min_field_freq]

    wordcount = Counter()
    with open(sent_path) as sent_f:
        sentences = sent_f.readlines()

    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > sum_seq_len:
            tokens = tokens[:sum_seq_len]
        wordcount.update(tokens)

    count_pairs = wordcount.most_common()
    count_pairs = wordcount.most_common(top_k - 5)
    words, _ = list(zip(*count_pairs))

    words = list(set(words) | set(flist))
    word_to_id = dict(zip(words, range(5, len(words) + 5)))

    word_to_id['<PAD>'] = 0
    word_to_id['<EOS>'] = 1
    word_to_id['<OOV>'] = 2
    word_to_id['<GO>'] = 3
    word_to_id['<COPY>'] = 4

    return word_to_id


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
        source.append(field)
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


def parse_input(box, sent, max_source_len, sum_seq_len, tokens_per_field,
                fields_per_box, word_to_id):
    # encoder inputs
    tokens = box.split()
    f2t = {}
    fields = []

    for token in tokens:
        field, value = token.split(':', 1)
        if value != '<none>' and value != ',' and value != '.':
            fname = field_name(field)
            if fname not in fields:
                if len(fields) < fields_per_box:
                    fields.append(fname)
                    f2t[fname] = [value]
            else:
                if len(f2t[fname]) < tokens_per_field:
                    f2t[fname].append(value)

    source = []
    for field in fields:
        source.append(field)
        source.extend(f2t[field])

    if len(source) > max_source_len:
        source = source[:max_source_len]
    else:
        source.extend(['<PAD>'] * (max_source_len - len(source)))

    box_in = [word_to_id.get(w, 2) for w in source]
    assert len(box_in) == max_source_len

    # decoder inputs
    len_vocab = len(word_to_id)
    swords = sent.split()
    swords = ['<GO>'] + swords
    num_valid = len(swords)
    z = []
    dec_in = []

    for sword in swords:
        if sword in source:
            z.append(1) # Copy
            dec_in.append(len_vocab + source.index(sword))
        else:
            z.append(0) # Choose from vocab
            dec_in.append(word_to_id.get(sword, 2))

    if len(dec_in) >= sum_seq_len + 1:
        dec_in = dec_in[:sum_seq_len]
        z = z[:sum_seq_len + 1]
        dec_in.extend([word_to_id.get('<EOS>')])
    else:
        dec_in.extend([word_to_id.get('<EOS>')])
        dec_in.extend([word_to_id.get('<PAD>')] *
                      (sum_seq_len + 1 - len(dec_in)))
        z.extend([0] * (sum_seq_len + 1 - len(z)))

    dec_wts = [1] * num_valid
    if num_valid < sum_seq_len:
        dec_wts.extend([0] * (sum_seq_len - num_valid))
    else:
        dec_wts = dec_wts[:sum_seq_len]

    return box_in, dec_in, z, dec_wts

class BaseDatasetCopy(object):
    """ BaseDatasetCopy : Defines the dataset object for the baseline model.

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
                 fields_per_box,
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
        self.z = []

        for (box, sent) in zip(self.boxes, self.sentences):
            box_in, dec_in, z, dec_wts = parse_input(box, sent,
                                                     max_source_len,
                                                     max_sum_len,
                                                     tokens_per_field,
                                                     fields_per_box,
                                                     word_to_id)
            self.input_box.append(box_in)
            self.decoder_inputs.append(dec_in)
            self.z.append(z)
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
            z_last = self.z[-1]

            for _ in range(extend):
                self.input_box.append(in_last)
                self.decoder_inputs.append(dec_in_last)
                self.decoder_weights.append(dec_wt_last)
                self.z.append(z_last)

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
        self.z = np.asarray(self.z, dtype=np.float32)

        self.batch_size = batch_size
        self._num_examples = len(self.sentences)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._permutation = np.random.permutation(self._num_batches)

    def next_batch(self):
        """ next_batch : Generate the next batch for the inference procedure.
        """
        batch_size = self.batch_size
        start = self._permutation[self._index_in_epoch] * batch_size
        end = start + batch_size
        self._index_in_epoch += 1

        if self._index_in_epoch == self._num_batches - 1:
            self._index_in_epoch = 0
            self._epochs_completed += 1
            self._permutation = np.random.permutation(self._num_batches)

        enc_inputs = self.input_box[start:end]
        dec_inputs = self.decoder_inputs[start:end]
        dec_weights = self.decoder_weights[start:end]
        z = self.z[start:end]

        return enc_inputs, dec_inputs, z, dec_weights

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

    def next_batch(self):
        """ next_batch : Generate the next batch for the inference procedure.
        """
        batch_size = self.batch_size
        #start = self._permutation[self._index_in_epoch] * batch_size
        start = self._index_in_epoch
        end = start + batch_size
        self._index_in_epoch = end

        if self._index_in_epoch == self._num_batches - 1:
            self._index_in_epoch = 0
            self._epochs_completed += 1

        enc_inputs = self.input_box[start:end]
        dec_inputs = self.decoder_inputs[start:end]
        dec_weights = self.decoder_weights[start:end]

        return enc_inputs, dec_inputs, dec_weights

    def next_batch_gen(self):
        """ next_batch_gen : Generate the next batch for generating sentence.
		difference from next_batch : returns true sentences in addtion to
		model inputs.
        """
        batch_size = self.batch_size
        #start = self._permutation[self._index_in_epoch] * batch_size
        start = self._index_in_epoch
        end = start + batch_size
        self._index_in_epoch = end

        if self._index_in_epoch == self._num_batches - 1:
            self._index_in_epoch = 0
            self._epochs_completed += 1

        enc_inputs = self.input_box[start:end]
        dec_inputs = self.decoder_inputs[start:end]
        dec_weights = self.decoder_weights[start:end]
        sents = self.sentences[start:end]

        return enc_inputs, dec_inputs, dec_weights, sents

    def reset_batch(self, epochs_completed=0):
        """ reset_batch : Reset the dataset. Equivalent to the
        condition at the start of the 1 epoch.
        """
        self._epochs_completed = epochs_completed
        self._index_in_epoch = 0

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
