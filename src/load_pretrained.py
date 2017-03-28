import numpy as np

def build_index(target_words, embed_path, top_k=20000):
    with open(target_words) as targ_f:
        lines = targ_f.readlines()

    lines = [line.rstrip() for line in lines]
    word_to_id = {}
    lines = lines[:top_k - 3]

    preidx = dict(zip(lines, range(top_k)))

    init_embed = np.genfromtxt(embed_path)
    dim = np.shape(init_embed)[1]
    init_embed[0] = np.zeros(dim)

    word_to_id['<PAD>'] = 0
    word_to_id['<EOS>'] = 1
    word_to_id['<OOV>'] = 2
    word_to_id['<GO>'] = 3
    word_to_id['<COPY>'] = 4


    init = np.zeros((top_k, dim))
    pad = np.zeros(dim)
    init[0] = pad
    init[1] = init_embed[preidx['<EOS>']]
    oov = np.random.randn(dim)
    init[2] = oov
    init[3] = init_embed[preidx['<GO>']]
    copy = np.random.randn(dim)
    init[4] = copy

    idx = 5
    for word in preidx:
        if word not in word_to_id:
            word_to_id[word] = idx
            init[idx] = init_embed[preidx[word]]
            idx += 1

    init = np.asarray(init, dtype=np.float32)

    return word_to_id, init
