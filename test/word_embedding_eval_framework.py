import copy
import fasttext
import io
import itertools
import logging
import numba
import numpy as np
import os
import random
import scipy.sparse
import scipy.stats
import senteval
import sys
import textmap
import textmap.tokenizers
import textmap.transformers
import time
import umap
import vectorizers
from enstop import EnsembleTopics, PLSA, BlockParallelPLSA
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from six import iteritems
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from web.datasets.similarity import fetch_WS353, fetch_RG65, fetch_RW, fetch_MTurk, fetch_MEN, fetch_SimLex999
from web.embedding import Embedding
from web.evaluate import evaluate_similarity
from web.vocabulary import Vocabulary

# ============================= PRELIMINARIES ============================= #

timestamp = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_tokens(path_to_txt_files):
    """
    :param path_to_txt_files: path to folder of txt files making up corpus
    :return: tokens
    """
    data = []
    dir = path_to_txt_files
    for file in os.listdir(dir):
        if file.endswith('.txt'):
            data.append(open(os.path.join(dir, file), 'r').read())
    tokens = textmap.tokenizers.SKLearnTokenizer(tokenize_by='sentence').fit_transform(data)
    return tokens


# ============================= EMBEDDING ============================= #

@numba.njit(parallel=True)
def ppmi_transform(row, col, val, row_marginal, col_marginal, alpha=0.75, shift=0.0):
    result_val = np.empty_like(val)
    for i in numba.prange(row.shape[0]):
        new_val = val[i]
        new_val /= (col_marginal[row[i]] * row_marginal[col[i]] ** alpha)
        if new_val > np.exp(-shift):
            new_val = np.log(new_val) + shift
        else:
            new_val = 0.0
        result_val[i] = new_val
    return result_val


def embed(tokens,
          skip_mte=False, max_token_frequency=1e-1, min_ngram_occurrences=50,
          max_frequency=1e-1, min_occurrences=50, window_radius=5, window_orientation='after',
          window_function='fixed', kernel_function='flat', skip_ppmi=True, alpha=0.75,
          skip_iwt=False, information_function='column_kl', binarize_matrix=False,
          skip_ret=False, em_background_prior=5, em_prior_strength=0.02, symmetrize=False,
          skip_dim_reduc=False, dim_reduc='plsa', n_components=100, l2_normalize=False, l1_normalize=False):
    """
    :return: new_tokens: tuple of sentences, represented as tuples of strings
             emb: dict of the form {word : word vector}
    """

    # ------ Multi-token expression transformer ------#
    logging.info("multi-token-expression-transforming")
    if skip_mte:
        new_tokens = tokens
    else:
        mte = textmap.transformers.MultiTokenExpressionTransformer(max_token_frequency=max_token_frequency,
                                                                   min_ngram_occurrences=min_ngram_occurrences,
                                                                   ignored_tokens=stopwords.words('english'),
                                                                   excluded_token_regex="\W+")
        new_tokens = mte.fit_transform(tokens)

    # ------ Token co-occurrence vectorizer ------#
    logging.info("token-co-occurrence-vectorizing")
    vectorizer = vectorizers.TokenCooccurrenceVectorizer(max_frequency=max_frequency,
                                                         min_occurrences=min_occurrences,
                                                         window_radius=window_radius,
                                                         window_orientation=window_orientation,
                                                         window_function=window_function,
                                                         kernel_function=kernel_function,
                                                         ignored_tokens=stopwords.words('english'))
    count_matrix = vectorizer.fit_transform(new_tokens)

    # ------ Positive pointwise mutual information ----#
    # logging.info("ppmi-transforming")
    # if skip_ppmi:
    #     ppmi_matrix, ppmi_matrix_t = count_matrix, count_matrix.T
    # else:
    #     row_marginal = np.array(count_matrix.sum(axis=0))[0]
    #     col_marginal = np.array(count_matrix.sum(axis=1)).T[0]
    #     ppmi_matrix = count_matrix.tocoo()
    #     ppmi_matrix_t = count_matrix.transpose().tocoo()
    #
    #     ppmi_matrix.data = ppmi_transform(ppmi_matrix.row, ppmi_matrix.col, ppmi_matrix.data,
    #                                       row_marginal, col_marginal, alpha=alpha, shift=5.0)
    #     ppmi_matrix = ppmi_matrix.tocsr()
    #     ppmi_matrix.eliminate_zeros()
    #
    #     ppmi_matrix_t.data = ppmi_transform(ppmi_matrix_t.row, ppmi_matrix_t.col, ppmi_matrix_t.data,
    #                                         col_marginal, row_marginal, alpha=alpha, shift=5.0)
    #     ppmi_matrix_t = ppmi_matrix_t.tocsr()
    #     ppmi_matrix_t.eliminate_zeros()

    # ------ Information weight transformer ------#
    logging.info("information-weight-transforming")
    if skip_iwt:
        info_mat, info_mat_t = count_matrix, count_matrix.T
    else:
        info_mat = textmap.transformers.InformationWeightTransformer(information_function=information_function,
                                                                     binarize_matrix=binarize_matrix
                                                                     ).fit_transform(count_matrix)
        info_mat_t = textmap.transformers.InformationWeightTransformer(information_function=information_function,
                                                                       binarize_matrix=binarize_matrix
                                                                       ).fit_transform(count_matrix.T)

    # ------ Remove effects transformer ------#
    logging.info("remove-effects-transforming")
    if skip_ret:
        re_mat, re_mat_t = info_mat, info_mat_t
    else:
        re_transformer = textmap.transformers.RemoveEffectsTransformer(em_background_prior=em_background_prior,
                                                                       em_prior_strength=em_prior_strength
                                                                       ).fit(info_mat)
        re_transformer_t = textmap.transformers.RemoveEffectsTransformer(em_background_prior=em_background_prior,
                                                                         em_prior_strength=em_prior_strength
                                                                         ).fit(info_mat_t)
        re_mat = re_transformer.transform(info_mat)
        re_mat_t = re_transformer_t.transform(info_mat_t)

    # ------ HStack ------#
    if symmetrize:
        logging.info("symmetrizing")
        hstack_mat = (re_mat + re_mat_t).tocsr()
    else:
        logging.info("hstacking")
        hstack_mat = scipy.sparse.hstack([re_mat, re_mat_t])

    # ------ Dimension reduction ------#
    if skip_dim_reduc:
        wvecs = hstack_mat.toarray()
    elif dim_reduc == 'svd':
        logging.info("svd-ing")
        model = TruncatedSVD(n_components=n_components, algorithm="arpack").fit(hstack_mat)
        wvecs = model.transform(hstack_mat)
    elif dim_reduc == 'plsa':
        logging.info("plsa-ing")
        model = BlockParallelPLSA(n_components=n_components, n_iter=300).fit(hstack_mat)
        wvecs = model.embedding_
    elif dim_reduc == 'enstop':
        logging.info("enstopping")
        model = EnsembleTopics(n_components=n_components).fit(hstack_mat)
        wvecs = model.embedding_
    elif dim_reduc == 'umap':
        logging.info("umapping")
        wvecs = umap.UMAP(n_components=n_components).fit_transform(hstack_mat.toarray())

    # ------ Normalisation ------#
    if l2_normalize:
        wvecs = normalize(wvecs, norm="l2")
    elif l1_normalize:
        wvecs = normalize(wvecs, norm="l1")

    # ------ Return embedding ------#
    emb = {}
    for w in vectorizer.column_label_dictionary_:
        emb[w] = wvecs[vectorizer.column_label_dictionary_[w]]
    return new_tokens, emb


def get_wordvec_(path_to_vec):
    """
    From file, returns dict of the form {word : word_vec}
    """
    word_vec = {}
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')
    return word_vec


def get_wordvec(path_to_vec, word2id):
    """
    :param path_to_vec: path to saved embedding
    :param word2id: dict of words and indices
    :return: wvecs: dict of words from word2id and corresponding word vectors
    """

    wvecs = {}
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                wvecs[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(wvecs), len(word2id)))
    return wvecs


def save_dict(emb, path_to_vec, dim=100):
    """
    save embedding to .txt file
    """
    word_vec = {}
    for w in emb.keys():
        word_vec[w] = emb[w]
    with open(path_to_vec + f'embedding_{str(dim)}d.txt', 'w') as file:
        for w, v in word_vec.items():
            file.write(w + ' ')
            for el in v:
                file.write("{:.20f}".format(float(el)) + ' ')
            file.write('\n')


def load_dict(path_to_vec):
    """
    load embedding from .txt file
    """
    emb = {}
    with open(path_to_vec, 'r', errors='ignore', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            emb[word] = vector
    return emb


# ============================= EVALUATION ============================= #

def create_dictionary(tokens):
    """
    :return: id2word: list of tokens, in decreasing order of appearances
             word2id: dict of words and indices
    """

    words = {}
    for s in tokens:
        for word in s:
            words[word] = words.get(word, 0) + 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def prepare(params, samples):
    """

    :param params: senteval parameters.
    :param samples: list of all sentences from the tranfer task.
    :return: No output. Arguments stored in "params" can further be used by batcher.
    """

    _, params['word2id'] = create_dictionary(samples)
    params['word_vec'] = get_wordvec(path_to_vec, params['word2id'])

    sentence_matrix = sklearn.feature_extraction.text.CountVectorizer(
        vocabulary=list(params["word_vec"].keys())
    ).fit_transform([" ".join(x) for x in samples])

    params["iwt"] = textmap.transformers.InformationWeightTransformer().fit(sentence_matrix)
    word_sentence_matrix = normalize(params["iwt"].transform(sentence_matrix), norm="l1") @ \
                           np.array(list(params["word_vec"].values()))
    params["re"] = textmap.transformers.RemoveEffectsTransformer().fit(scipy.sparse.csr_matrix(word_sentence_matrix))
    re_matrix = params["re"].transform(scipy.sparse.csr_matrix(word_sentence_matrix)).toarray()
    params["scaler"] = sklearn.preprocessing.StandardScaler().fit(re_matrix)
    return


def batcher(params, batch):
    sentence_matrix = sklearn.feature_extraction.text.CountVectorizer(
        vocabulary=list(params["word_vec"].keys())
    ).fit_transform([" ".join(x) for x in batch])
    re_matrix = normalize(sentence_matrix, norm="l1") @ np.array(list(params["word_vec"].values()))
    return params["scaler"].transform(re_matrix)


def web_tests(emb):
    """
    :param emb: dict of words and their corresponding embeddings
    :return: dict of word-embeddings-benchmarks tests and scores received
    """
    similarity_tasks = {'WS353': fetch_WS353(), 'RG65': fetch_RG65(), 'RW': fetch_RW(),
                        'MTurk': fetch_MTurk(), 'MEN': fetch_MEN(), 'SimLex999': fetch_SimLex999()}

    web_emb = Embedding(Vocabulary(list(emb.keys())), list(emb.values()))
    similarity_results = {}
    for name, data in iteritems(similarity_tasks):
        similarity_results[name] = evaluate_similarity(web_emb, data.X, data.y)
        logging.info(
            "Spearman correlation of scores on {} {}".format(name, evaluate_similarity(web_emb, data.X, data.y)))
    return similarity_results


# ----------------------------- TESTS ----------------------------- #

def textmap_test(tokens,
                 dir='./test_results_{timestamp}.txt', path_to_vec='.',
                 skip_mte=False, max_token_frequency=1e-1, min_ngram_occurrences=50,
                 max_frequency=1e-1, min_occurrences=50, window_radius=5, window_orientation='after',
                 window_function='fixed', kernel_function='flat', skip_ppmi=True, alpha=0.75,
                 skip_iwt=False, information_function='column_kl', binarize_matrix=False,
                 skip_ret=False, em_background_prior=5, em_prior_strength=0.02, symmetrize=False,
                 skip_dim_reduc=False, dim_reduc='plsa', n_components=100, l2_normalize=False, l1_normalize=False,
                 transfer_tasks=None):
    # ------ Set-up ------#
    if transfer_tasks is None:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'SNLI', 'TREC', 'MRPC', 'SICKEntailment']

    param_pairs = [('skip_mte', skip_mte), ('max_token_frequency', max_token_frequency),
                   ('min_ngram_occurrences', min_ngram_occurrences),
                   ('max_frequency', max_frequency), ('min_occurrences', min_occurrences),
                   ('window_radius', window_radius), ('window_orientation', window_orientation),
                   ('window_function', window_function), ('kernel_function', kernel_function),
                   ('skip_iwt', skip_iwt), ('information_function', information_function),
                   ('binarize_matrix', binarize_matrix),
                   ('skip_ret', skip_ret), ('em_background_prior', em_background_prior),
                   ('em_prior_strength', em_prior_strength),
                   ('skip_dim_reduc', skip_dim_reduc), ('dim_reduc', dim_reduc), ('n_components', n_components)]

    with open(dir, 'a') as file:
        for par_str, par in param_pairs:
            file.write(par_str + ' = ' + str(par) + ' | ')
        file.write('\n\n')

    # ------ Embedding ------#
    logging.info("embedding")
    new_tokens, emb = embed(tokens,
                            skip_mte=skip_mte, max_token_frequency=max_token_frequency,
                            min_ngram_occurrences=min_ngram_occurrences,
                            max_frequency=max_frequency, min_occurrences=min_occurrences,
                            window_radius=window_radius,
                            window_orientation=window_orientation,
                            window_function=window_function, kernel_function=kernel_function,
                            skip_ppmi=skip_ppmi, alpha=alpha,
                            skip_iwt=skip_iwt, information_function=information_function,
                            binarize_matrix=binarize_matrix,
                            skip_ret=skip_ret, em_background_prior=em_background_prior,
                            em_prior_strength=em_prior_strength, symmetrize=symmetrize,
                            skip_dim_reduc=skip_dim_reduc, dim_reduc=dim_reduc, n_components=n_components,
                            l2_normalize=l2_normalize, l1_normalize=l1_normalize)
    save_dict(emb, path_to_vec=path_to_vec + f'embedding_{str(n_components)}d.txt')

    # ------ WEB ------#
    logging.info("obtaining word-embeddings-benchmarks results")
    web_results = web_tests(emb)

    logging.info("word-embeddings-benchmarks results: " + str(web_results))
    with open(dir, 'a') as file:
        file.write(str(web_results) + '\n\n')

    # ------ SentEval ------#
    logging.info("obtaining senteval results")
    params = {'task_path': 'PATH_TO_SENTEVAL/data',
              'usepytorch': False, 'kfold': 10}

    se = senteval.engine.SE(params, batcher, prepare)
    se_results = se.eval(transfer_tasks)

    logging.info("senteval results: " + str(se_results))
    with open(dir, 'a') as file:
        file.write(str(se_results) + '\n\n----------------------------------\n\n')


def w2v_embed(tokens, max_token_frequency=1e-1, min_ngram_occurrences=50,
              size=100, window=5, min_count=50, iter=10, progress_per=1000):
    """
    :param tokens: 
    :param max_token_frequency: 
    :param min_ngram_occurrences: 
    :param size: Dimensionality of the word vectors.
    :param window: Maximum distance between the current and predicted word within a sentence.
    :param min_count: Ignores all words with total frequency lower than this.
    :param iter: Number of iterations (epochs) over the corpus.
    :param progress_per: Indicates how many words to process before showing/updating the progress.
    """

    mte = textmap.transformers.MultiTokenExpressionTransformer(max_token_frequency=max_token_frequency,
                                                               min_ngram_occurrences=min_ngram_occurrences,
                                                               ignored_tokens=stopwords.words('english'),
                                                               excluded_token_regex="\W+")
    new_tokens = mte.fit_transform(tokens)
    model = Word2Vec(size=size, window=window, min_count=min_count, iter=iter)
    model.build_vocab(new_tokens, progress_per=progress_per)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    model.init_sims(replace=True)

    emb = {}
    for w in model.wv.vocab:
        emb[w] = model.wv[w]
    return new_tokens, emb, model


def w2v_test(tokens, dir=f'./test_results_{timestamp}.txt', path_to_vec='.',
             max_token_frequency=1e-1, min_ngram_occurrences=50,
             size=100, window=5, min_count=50, iter=10, progress_per=10000,
             transfer_tasks=None):
    """
    :param size: Dimensionality of the word vectors.
    :param window: Maximum distance between the current and predicted word within a sentence.
    :param min_count: Ignores all words with total frequency lower than this.
    :param iter: Number of iterations (epochs) over the corpus.
    :param progress_per: Indicates how many words to process before showing/updating the progress.
    """

    if transfer_tasks is None:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'SNLI', 'TREC', 'MRPC', 'SICKEntailment']

    param_pairs = [('method', 'word2vec'),
                   ('max_token_frequency', max_token_frequency), ('min_ngram_occurrences', min_ngram_occurrences),
                   ('size', size), ('window', window), ('min_count', min_count), ('iter', iter)]

    with open(dir, 'a') as file:
        for par_str, par in param_pairs:
            file.write(par_str + ' = ' + str(par) + ' | ')
        file.write('\n\n')

    # ------ Embedding ------#
    logging.info("preparing word embedding")
    _, emb, _ = w2v_embed(tokens,
                          max_token_frequency=max_token_frequency, min_ngram_occurrences=min_ngram_occurrences,
                          size=size, window=window, min_count=min_count, iter=iter, progress_per=progress_per)
    save_dict(emb, path_to_vec=path_to_vec + f'embedding_{str(size)}d.txt')

    # ------ WEB ------#
    logging.info("getting WEB results")
    web_results = web_tests(emb)

    logging.info("WEB results: " + str(web_results))
    with open(dir, 'a') as file:
        file.write(str(web_results) + '\n\n')

    # ------ SentEval ------#
    logging.info("getting SentEval results")
    params = {'task_path': 'PATH_TO_SENTEVAL/data', 'usepytorch': False, 'kfold': 10}

    se = senteval.engine.SE(params, batcher, prepare)
    se_results = se.eval(transfer_tasks)

    logging.info("SentEval results: " + str(se_results))
    with open(dir, 'a') as file:
        file.write(str(se_results) + '\n\n----------------------------------\n\n')


def prepare_glove_txt(tokens, max_frequency=1e-1, min_occurrences=50,
                      corpus_path='./corpus.txt', vocab_path='PATH_TO_GLOVE/vocab.txt'):
    mte = textmap.transformers.MultiTokenExpressionTransformer(max_token_frequency=max_frequency,
                                                               min_ngram_occurrences=min_occurrences,
                                                               ignored_tokens=stopwords.words('english'),
                                                               excluded_token_regex="\W+")
    tokens = mte.fit_transform(tokens)

    flat_tokens = vectorizers.utils.flatten(tokens)
    (result_sequences, token_dictionary, inverse_token_dictionary,
     token_frequencies) = vectorizers._vectorizers.preprocess_token_sequences(tokens, flat_tokens,
                                                                              max_frequency=max_frequency,
                                                                              min_occurrences=min_occurrences,
                                                                              ignored_tokens=stopwords.words('english'),
                                                                              excluded_token_regex="\W+")
    open(corpus_path, 'w')
    words = {}
    for sent in result_sequences:
        for w_no in sent:
            words[inverse_token_dictionary[w_no]] = words.get(inverse_token_dictionary[w_no], 0) + 1
            with open(corpus_path, 'a') as file:
                file.write(inverse_token_dictionary[w_no] + ' ')

    with open(vocab_path, 'w') as file:
        for w in words:
            file.write(w + ' ' + str(words[w]) + '\n')


def fasttext_test(tokens, fasttext_model='skipgram',
                  max_frequency=1e-1, min_occurrences=50, dim=100, window=5, epoch=15,
                  dir=f'./test_results_{timestamp}.txt',
                  corpus_path='./corpus.txt', vocab_path='PATH_TO_GLOVE/vocab.txt', create_txt=False,
                  transfer_tasks=None
                  ):
    """
    :param fasttext_model: unsupervised fasttext model (cbow, skipgram)
    :param dim: size of word vectors
    :param window: size of the context window
    :param epoch: number of epochs
    :return: 
    """

    # ------ set-up ------#
    if transfer_tasks is None:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'SNLI', 'TREC', 'MRPC', 'SICKEntailment']

    param_pairs = [('fasttext_model', fasttext_model),
                   ('max_frequency', max_frequency), ('min_occurrences', min_occurrences),
                   ('dim', 100), ('window', window), ('epoch', epoch)]

    with open(dir, 'a') as file:
        for par_str, par in param_pairs:
            file.write(par_str + ' = ' + str(par) + ' | ')
        file.write('\n\n')

    # ------ Embedding ------#
    logging.info("preparing word embedding")

    if create_txt:
        prepare_glove_txt(tokens, max_frequency=max_frequency, min_occurrences=min_occurrences,
                          corpus_path=corpus_path, vocab_path=vocab_path)

    model = fasttext.train_unsupervised(corpus_path,
                                        model=fasttext_model, dim=dim, ws=window, epoch=epoch)

    emb = {}
    for w in model.words:
        emb[w] = model[w]

    # ------ WEB ------#
    logging.info("getting WEB results")
    web_results = web_tests(emb)

    logging.info("WEB results: " + str(web_results))
    with open(dir, 'a') as file:
        file.write(str(web_results) + '\n\n')

    # ------ SentEval ------#
    logging.info("getting SentEval results")
    params = {'task_path': 'PATH_TO_SENTEVAL/data', 'usepytorch': False, 'kfold': 10}

    se = senteval.engine.SE(params, batcher, prepare)
    se_results = se.eval(transfer_tasks)

    logging.info("SentEval results: " + str(se_results))
    with open(dir, 'a') as file:
        file.write(str(se_results) + '\n\n----------------------------------\n\n')


def test_glove(vectors_file='PATH_TO_GLOVE/vectors.txt', dir=f'./test_results_{timestamp}.txt', path_to_vec='.',
               max_frequency=1e-1, min_occurrences=50,
               verbose=2, memory=4.0, vocab_min_count=50, vector_size=100, max_iter=15,
               window_size=5, binary=2, num_threads=8, x_max=10,
               transfer_tasks=None):
    # ------ Set-up ------#
    logging.info("setting up")
    if transfer_tasks is None:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'SNLI', 'TREC', 'MRPC', 'SICKEntailment']

    with open(dir, 'a') as file:
        param_pairs = [('max_frequency', max_frequency), ('min_occurrences', min_occurrences),
                       ('VERBOSE', verbose), ('memory', memory), ('vocab_min_count', vocab_min_count),
                       ('vector_size', vector_size), ('max_iter', max_iter), ('window_size', window_size),
                       ('binary', binary), ('num_threads', num_threads), ('x_max', x_max)]
        for par_str, par in param_pairs:
            file.write(par_str + ' = ' + str(par) + ' | ')
        file.write('\n\n')

    emb = get_wordvec_(vectors_file)
    save_dict(emb, path_to_vec=path_to_vec + f'embedding_{str(vector_size)}d.txt')

    # ------ WEB ------#
    logging.info("obtaining word-embeddings-benchmarks results")
    web_results = web_tests(emb)

    logging.info("word-embeddings-benchmarks results: " + str(web_results))
    with open(dir, 'a') as file:
        file.write(str(web_results) + '\n\n')

    # ------ SentEval ------#
    logging.info("obtaining senteval results")
    params = {'task_path': 'PATH_TO_SENTEVAL/data', 'usepytorch': False, 'kfold': 10}

    se = senteval.engine.SE(params, batcher, prepare)
    se_results = se.eval(transfer_tasks)

    logging.info("senteval results: " + str(se_results))
    with open(dir, 'a') as file:
        file.write(str(se_results) + '\n\n----------------------------------\n\n')


# ============================= SYNONYM AUGMENTATION ============================= #

def synonym_token_replace(tokens, ignored_tokens=stopwords.words('english'), excluded_token_regex=None,
                          max_frequency=None, min_occurrences=None,
                          # min_frequency=None,  max_occurrences=None,
                          # min_document_frequency=None, max_document_frequency=None,
                          # min_document_occurrences=None, max_document_occurrences=None,
                          num_candidates=25, replace_probability=0.5, tokens_to_replace=None):
    """
    Iterate over documents, checking probabilities for token replacement with synonyms.

    Based on user-defined probabilities, candidate tokens will be replaced with one of k synonyms.
    These synonyms will take the form <original_token>_$$<k> where k is the index of the kth probability
    passed by the user. A new sequence of tokenized documents, including created synonyms, will be built.

    :param tokens: a tuple of tuples of tokenized documents
    :param ignored_tokens: a set of tokens to prune from token dictionary
    :param excluded_token_regex: a regex pattern to identify tokens to prune from token dictionary
    #:param min_frequency: float - The minimum frequency of occurrence allowed for tokens. Tokens that occur
    #    less frequently than this will be pruned.
    :param max_frequency: float - The maximum frequency of occurrence allowed for tokens. Tokens that occur
        more frequently than this will be pruned.
    :param min_occurrences: int - A constraint on the minimum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.
    #:param max_occurrences: int - A constraint on the maximum number of occurrences for a token to be considered
    #    valid. If None then no constraint will be applied.
    #:param min_document_frequency: int - A constraint on the minimum frequency of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param max_document_frequency: int - A constraint on the maximum frequency of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param min_document_occurrences: int - A constraint on the minimum number of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param max_document_occurrences: int - A constraint on the maximum number of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    :param num_candidates: int - The number of candidate tokens to be replaced with synonyms
    :param tokens_to_replace: list - A list of tokens to be replaced with synonyms.  If None, the other parameters
    :param replace_probability: list - List of floats of probabilities for synonym creation
    :return:  a tuple of tuples of tokenized documents containing new synonym in place of original tokens
    """

    if not tokens_to_replace:
        # flatten tuple of tuples and get token dictionary
        token_dict, token_freq, n_tokens = vectorizers._vectorizers.construct_token_dictionary_and_frequency(
            vectorizers.utils.flatten(tokens))

        # prune token dictionary depending on parameters supplied by user
        # returns a dictionary of candidate tokens for replacement
        candidate_dict, candidate_freq = vectorizers._vectorizers.prune_token_dictionary(
            token_dict,
            token_freq,
            ignored_tokens=ignored_tokens,
            excluded_token_regex=excluded_token_regex,
            min_frequency=(min_occurrences / n_tokens),
            max_frequency=max_frequency,
            # min_occurrences=min_occurrences,
            # max_occurrences=max_occurrences,
            # min_document_frequency=min_document_frequency,
            # max_document_frequency=max_document_frequency,
            # min_document_occurrences=min_document_occurrences,
            # max_document_occurrences=max_document_occurrences,
            total_tokens=n_tokens,
            total_documents=len(tokens),
        )

        # take a random sample of tokens from the candidate dictionary
        tokens_to_replace = random.sample(list(candidate_dict.keys()), num_candidates)

    print("Tokens for replacement:")
    print(tokens_to_replace)

    # normalize replacement_probability
    norm_prob = np.array([replace_probability, 1 - replace_probability]).reshape(1, -1)
    norm_prob = normalize(norm_prob, axis=1, norm='l1').flatten().tolist()

    new_doc_list = []
    for doc in tokens:
        new_doc = []
        for token in doc:
            if token not in tokens_to_replace:
                new_doc.append(token)  # new_doc.append(f"{token}_$$orig")
            else:
                synonyms = []
                for idx, _ in enumerate(norm_prob):
                    synonyms.append(f"{token}_$${idx}")
                synonym = np.random.choice(synonyms, p=norm_prob)
                # logging.info("replacing '{}' with '{}'".format(token,synonym)) # print(synonym)
                new_doc.append(str(synonym))
        new_doc_list.append(new_doc)

    # change dataset back to tuple of tuples before returning
    new_doc_tuple = tuple(tuple(doc) for doc in new_doc_list)
    return tokens_to_replace, new_doc_tuple


def synonym_sentence_append(tokens, ignored_tokens=stopwords.words('english'), excluded_token_regex=None,
                            max_frequency=None, min_occurrences=None,
                            # min_frequency=None, max_occurrences=None,
                            # min_document_frequency=None, max_document_frequency=None,
                            # min_document_occurrences=None, max_document_occurrences=None,
                            num_candidates=25, tokens_to_replace=None, replace_probability=0.3):
    """
    Take a tuple of tokenized sentences.  For a list of candidate tokens, replaces a single instance of that token
    in a sentence with <original_token>_$$0 and adding that sentence to a new corpus of tuple of sentences.  If a
    sentence contains multiple candidate tokens, only one will be replaced at a time and multiple copies of the
    sentence, each with a single replaced token, will be added to the new corpus.

    When a sentence with one of these manufactured synonms is added to the new corpus, an identical copy of the
    sentence may again be added based on a user provided probability, but with a different manufactured synonym,
    <original_token>_$$1. Finally, if sentence contains no candidate tokens, it is added to the new corpus unchanged.

    :param tokens: a tuple of tuples of tokenized sentences
    :param ignored_tokens: a set of tokens to prune from token dictionary
    :param excluded_token_regex: a regex pattern to identify tokens to prune from token dictionary
    #:param min_frequency: float - The minimum frequency of occurrence allowed for tokens. Tokens that occur
    #    less frequently than this will be pruned.
    :param max_frequency: float - The maximum frequency of occurrence allowed for tokens. Tokens that occur
        more frequently than this will be pruned.
    :param min_occurrences: int - A constraint on the minimum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.
    #:param max_occurrences: int - A constraint on the maximum number of occurrences for a token to be considered
    #    valid. If None then no constraint will be applied.
    #:param min_document_frequency: int - A constraint on the minimum frequency of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param max_document_frequency: int - A constraint on the maximum frequency of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param min_document_occurrences: int - A constraint on the minimum number of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    #:param max_document_occurrences: int - A constraint on the maximum number of documents with occurrences for a
    #    token to be considered valid. If None then no constraint will be applied.
    :param num_candidates: int - The number of candidate tokens to be replaced with synonyms
    :param tokens_to_replace: list - A list of tokens to be replaced with synonyms.  If None, the other parameters
        will be used to select tokens to replace
    :param replace_probability: float - the probability a new synonym sentence will be added to the corpus
    :return:  a tuple of tuples of tokenized sentences containing new synonym in place of original tokens and
        a list of the words that were replaced
    """

    # check if tokens to be replaced are supplied, if not, choose tokens depending on parameters from user
    if not tokens_to_replace:
        # flatten tuple of tuples and get token dictionary
        token_dict, token_freq, n_tokens = vectorizers._vectorizers.construct_token_dictionary_and_frequency(
            vectorizers.utils.flatten(tokens))

        # prune token dictionary depending on parameters supplied by user
        # returns a dictionary of candidate tokens for replacement
        candidate_dict, candidate_freq = vectorizers._vectorizers.prune_token_dictionary(
            token_dict,
            token_freq,
            ignored_tokens=ignored_tokens,
            excluded_token_regex=excluded_token_regex,
            min_frequency=(min_occurrences / n_tokens),
            max_frequency=max_frequency,
            # min_occurrences=min_occurrences,
            # max_occurrences=max_occurrences,
            # min_document_frequency=min_document_frequency,
            # max_document_frequency=max_document_frequency,
            # min_document_occurrences=min_document_occurrences,
            # max_document_occurrences=max_document_occurrences,
            total_tokens=n_tokens,
            total_documents=len(tokens),
        )

        # take a random sample of tokens from the candidate dictionary
        tokens_to_replace = random.sample(list(candidate_dict.keys()), num_candidates)

    print("Tokens for replacement:")
    print(tokens_to_replace)

    new_tokens = []
    for sent in tokens:
        word_changed = False
        sent = list(sent)
        # check each token by index and create a deep copy with the changed word at that index and add new sentence
        # to new corpus
        for idx, token in enumerate(sent):
            if token in tokens_to_replace:
                new_sent = copy.deepcopy(sent)
                new_sent[idx] = f"{token}_$$0"
                new_tokens.append(new_sent)
                word_changed = True
                # depending on probability, add another copy of the new sentence with the second replacement synonym
                if random.random() <= replace_probability:
                    added_sent = copy.deepcopy(sent)
                    added_sent[idx] = f"{token}_$$1"
                    new_tokens.append(added_sent)
        # if no words were changed, just add the original sentence to the new corpus
        if not word_changed:
            new_tokens.append(sent)

    # change dataset back to tuple of tuples before returning
    new_tokens_tuple = tuple(tuple(sent) for sent in new_tokens)

    return tokens_to_replace, new_tokens_tuple


def synonym_test(tokens, alg='synonym_token_replace',
                 ignored_tokens=stopwords.words('english'),
                 num_candidates=100, replace_probability=0.5, tokens_to_replace=None,
                 skip_mte=False,
                 max_frequency=1e-1, min_occurrences=50,
                 window_radius=5, window_orientation='after', window_function='fixed', kernel_function='flat',
                 skip_ppmi=True, alpha=0.75,
                 skip_iwt=False, information_function='column_kl', binarize_matrix=False,
                 skip_ret=False, em_background_prior=5, em_prior_strength=0.02,
                 symmetrize=False,
                 skip_dim_reduc=False, dim_reduc='plsa', n_components=100,
                 l2_normalize=False, l1_normalize=False,):

    # ------ Multi-token expression transformer ------#
    if skip_mte:
        new_tokens = tokens
    else:
        mte = textmap.transformers.MultiTokenExpressionTransformer(max_token_frequency=max_frequency,
                                                                   min_ngram_occurrences=min_occurrences,
                                                                   ignored_tokens=ignored_tokens,
                                                                   excluded_token_regex="\W+")
        new_tokens = mte.fit_transform(tokens)

    # ------- Create synonym tokens -------#
    if alg == 'synonym_token_replace':
        tokens_to_replace, syn_tokens = synonym_token_replace(new_tokens, ignored_tokens=ignored_tokens,
                                                              min_occurrences=2 * min_occurrences,
                                                              max_frequency=max_frequency,
                                                              num_candidates=num_candidates,
                                                              replace_probability=replace_probability,
                                                              tokens_to_replace=tokens_to_replace)
    elif alg == 'synonym_sentence_append':
        tokens_to_replace, syn_tokens = synonym_sentence_append(new_tokens, ignored_tokens=ignored_tokens,
                                                                min_occurrences=2 * min_occurrences,
                                                                max_frequency=max_frequency,
                                                                num_candidates=num_candidates,
                                                                replace_probability=replace_probability,
                                                                tokens_to_replace=tokens_to_replace)

    # ------- Embedding -------#
    syn_tokens, emb = embed(syn_tokens,
                            skip_mte=True,
                            min_occurrences=None, max_frequency=None,
                            max_token_frequency=None, min_ngram_occurrences=None,
                            window_radius=window_radius, window_orientation=window_orientation,
                            window_function=window_function, kernel_function=kernel_function,
                            skip_ppmi=skip_ppmi, alpha=alpha,
                            skip_iwt=skip_iwt, information_function=information_function,
                            binarize_matrix=binarize_matrix,
                            skip_ret=skip_ret, em_background_prior=em_background_prior,
                            em_prior_strength=em_prior_strength, symmetrize=symmetrize,
                            skip_dim_reduc=skip_dim_reduc, dim_reduc=dim_reduc, n_components=n_components,
                            l2_normalize=l2_normalize, l1_normalize=l1_normalize)

    mean_dist = np.mean([umap.distances.hellinger(emb[f"{w}_$$0"], emb[f"{w}_$$1"]) for w in tokens_to_replace])

    # ------- Return -------#
    return tokens_to_replace, syn_tokens, emb, mean_dist
