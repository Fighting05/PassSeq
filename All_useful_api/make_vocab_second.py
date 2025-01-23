import collections
import pickle as pkl


def get_tokens(path):
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip('\n').split(' ')
            #print('1',tokens )
            tokens = [t.strip() for t in tokens]  # strip
            #print('2', tokens)
            tokens = [t for t in tokens if t != '']  # remove empty
            #print('3', tokens)
            yield tokens


def make_vocab(src_path, tgt_path, vocab_path):
    vocab_counter = collections.Counter()

    print("Reading sentences...")
    for tokens in get_tokens(src_path):
        vocab_counter.update(tokens)
    for tokens in get_tokens(tgt_path):
        vocab_counter.update(tokens)

    # saving
    print("Writing vocab file...")
    print('Print the vocab_counter!')
    print(vocab_counter)
    with open(vocab_path, 'wb') as f:
        pkl.dump(vocab_counter, f)
    print("Finished writing vacab file\n")


#Just run it!
if __name__ == "__main__":
    SRC_PATH = "../data//input.txt"
    TARGET_PATH = "../data/output.txt"
    VOCAB_PATH = '../data/vocab.pkl'
    make_vocab(SRC_PATH, TARGET_PATH, VOCAB_PATH)
