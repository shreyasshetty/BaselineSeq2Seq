""" vocabulary.py
    Contains functions for vocabulary building and
    processing.

    Reference for the additional vocabulary symbols.
    <PAD>  --  0
    <EOS>  --  1
    <OOV>  --  2
    <GO>   --  3
"""
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


def build_vocabulary(sent_path, top_k, max_sum_seq_len):
    """ build a vocabulary index of words in the dataset.

    Args:
        sent_path : The path to file containing sentences to be indexed.
        top_k     : Choose the top_k most frequent words to index the
                    dictionary.
        max_sum_seq_len : Max length of sentences to consider.

    Returns:
        word_to_id - Word to id mappings.
    """
    wordcount = Counter()
    with open(sent_path) as sent_f:
        sentences = sent_f.readlines()

    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > max_sum_seq_len:
            tokens = tokens[:max_sum_seq_len]
        wordcount.update(tokens)

    print "Words in the vocabulary : %d" % len(wordcount)

    count_pairs = wordcount.most_common()
    count_pairs = wordcount.most_common(top_k - 4)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(4, len(words) + 4)))

    word_to_id['<PAD>'] = 0
    word_to_id['<EOS>'] = 1
    word_to_id['<OOV>'] = 2
    word_to_id['<GO>'] = 3

    return word_to_id


def field_vocabulary(infoboxes, fields_per_table, min_field_freq):
    """ Generate the field vocabulary.

    Only consider fields_per_table many fields for building the index.

    Return: field_to_id
    """
    fieldcount = Counter()

    with open(infoboxes) as box_f:
        boxes = box_f.readlines()

    for box in boxes:
        tokens = box.split()
        fields = []
        for token in tokens:
            field, value = token.split(':', 1)
            if value != '<none>' and value != ',' and value != '.':
                fname = field_name(field)
                if fname not in fields:
                    if len(fields) == fields_per_table:
                        break
                    else:
                        fields.append(fname)
        fieldcount.update(fields)

    count_pairs = [c for c in fieldcount.most_common()
                   if c[1] >= min_field_freq]
    fields, _ = list(zip(*count_pairs))
    field_to_id = dict(zip(fields, range(2, len(fields) + 2)))

    field_to_id['<PAD>'] = 0
    field_to_id['<OOV>'] = 1

    return field_to_id


def preprocess(infoboxes, fields_per_box, tokens_per_field):
    with open(infoboxes) as info_f:
        boxes = info_f.readlines()

    tokens = []
    fields = []

    for box in boxes:
        toks = box.split()
        fs = []
        ts = []
        f2t = {}
        for tok in toks:
            key, value = tok.split(':', 1)
            if value != '<none>' and value != ',' and value != '.':
                fname = field_name(key)
                if fname in fs:
                    if len(f2t[fname]) < tokens_per_field:
                        f2t[fname].append(value)
                else:
                    if len(fs) < fields_per_box:
                        fs.append(fname)
                        f2t[fname] = [value]

        fields.append(fs)
        for f in fs:
            ts.append(f2t[f])
        tokens.append(ts)

    return tokens, fields


def tokens_to_ids(tok_list, word_to_id, tokens_per_field):
    seq_len = len(tok_list)

    tokens = [word_to_id.get(word, 2) for word in tok_list]

    if seq_len < tokens_per_field:
        tokens.extend([word_to_id.get('<PAD>')] * (tokens_per_field - seq_len))

    return seq_len, tokens


def fields_to_ids(field_list, field_to_id, fields_per_box):
    seq_len = len(field_list)

    fields = [field_to_id.get(field, 1) for field in field_list]

    if seq_len < fields_per_box:
        fields.extend([field_to_id.get('<PAD>')] * (fields_per_box - seq_len))

    return seq_len, fields


def process_tokens_fields(tokens, fields, word_to_id, field_to_id,
                          fields_per_box, tokens_per_field):
    """ tokens - list of list of tokens
        fields - list of fields
    """
    tokens_len = []
    fields_len = []
    tokens_in = []
    fields_in = []
    for tok_list in tokens:
        seq_len, toks = tokens_to_ids(tok_list, word_to_id, tokens_per_field)
        tokens_len.append(seq_len)
        tokens_in.append(toks)

    seq_field, ftoks = fields_to_ids(fields, field_to_id, fields_per_box)

    # If fewer fields than fields_per_box
    if seq_field < fields_per_box:
        tokens_len.extend([0] * (fields_per_box - seq_field))
        while len(tokens_in) < fields_per_box:
            tokens_in.append([word_to_id.get('<PAD>')] * tokens_per_field)

    fields_len.append(seq_field)
    fields_in.append(ftoks)

    return tokens_in, tokens_len, fields_in, fields_len
