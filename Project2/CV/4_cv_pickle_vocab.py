#!/usr/bin/env python3
import pickle

# Training 
def main():
    vocab = dict()
    with open('vocab_cut_cv2.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab_cv2.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

# Testing
def main():
    vocab = dict()
    with open('test_vocab_cut_cv2.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('test_vocab_cv2.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
