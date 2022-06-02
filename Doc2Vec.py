import gzip
import argparse
import os
import gensim
import time
import json
import multiprocessing

def tokenizer(sentence):
    s = sentence.replace('"','').replace("\\", " ")
    s = s.split(" ")
    while '' in s:
        s.remove('')
    while None in s :
        s.remove(None)
    return s

def collect_corpus(filelist, outputfile, wanted_type='PROCESS'):
    """

    :param filelist: List of filenames that we use to build the corpus
    :param outputfile: Path to the output file, where the final corpus will be saved
    :param wanted_type: Object type to keep
    :return: None
    """
    print('Collecting corpus...')
    f = open(outputfile, 'w')
    for file in filelist :
        h = gzip.open(file, 'rt')
        for l, line in enumerate(h):

            val_line = json.loads(line)
            #val_line = json.loads(val_line)

            otype = val_line['object']

            if otype == wanted_type:
                prop = val_line['properties']
                if 'command_line' in prop :
                    cmd = prop['command_line']
                    cmd = tokenizer(cmd)
                    s = ' '.join(cmd)
                    f.write(s + "\n")


def main(params):

    rootpath = params.rootpath
    corpus_savepath = params.corpus_savepath
    vector_size = params.vector_size
    n_epochs = params.n_epochs
    seed = params.seed
    model_savepath = params.model_savepath

    paths = [os.path.join(path, name) for path, subdirs, files in os.walk(rootpath) for name in files] #List of all the benign optc file (unsorted)
    collect_corpus(paths, corpus_savepath)

    model = gensim.models.doc2vec.Doc2Vec(corpus_file = corpus_savepath, vector_size=vector_size, min_count=1, epochs=n_epochs, seed=seed, workers = multiprocessing.cpu_count() - 1)
    print('Building vocabulary...')
    model.build_vocab(corpus_file = corpus_savepath)
    print('Training...')
    tic = time.time()
    model.train(corpus_file = corpus_savepath, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=model.epochs)
    tac = time.time()
    print('Done in', tac-tic)
    model.save(model_savepath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Doc2Vec')
    parser.add_argument('--rootpath', default='C:/Users/roxan/Documents/Dauphine/M2/Stage/optc/ecar/benign', help='Root filepath') #
    parser.add_argument('--corpus_savepath', default='C:/Users/roxan/Documents/Dauphine/M2/Stage/corpus.txt', help='Path where to save the corpus')
    parser.add_argument('--vector_size', default = 64, help='Vector_size for the Doc2Vec model')
    parser.add_argument('--seed', default=0, help='Seed for the random number generator')
    parser.add_argument('--model_savepath', default='C:/Users/roxan/Documents/Dauphine/M2/Stage/model', help='Path where the trained model should be saved')
    parser.add_argument('--n_epochs', default = 10, help='Number of epochs for the Doc2Vec model')
    args = parser.parse_args()

    main(args)


