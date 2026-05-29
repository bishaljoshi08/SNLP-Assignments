import string
from typing import List, Set, Tuple
from collections import Counter
import math


def preprocess_tokenized_corpus(corpus: List[List[str]], max_ngram_order: int = 10) -> List[List[str]]:
    '''
    Cleans the tokenized (unigram tokens) corpus.
    '''
    cleaned_corpus = []
    for sent in corpus:
        # Remove punctuations and lowercase the words
        translator = str.maketrans('', '', string.punctuation)
        cleaned_words = []
        for word in sent:
           cleaned = word.translate(translator).lower()
           if cleaned:
                cleaned_words.append(cleaned)

        # Remove sentences with less words than our highest ngram order
        if len(cleaned_words)>=max_ngram_order:
            cleaned_corpus.append(cleaned_words)

    return cleaned_corpus


def make_vocab(corpus: List[List[str]], top_k: int) -> Set[str]:
    '''
    Build the vocabulary set using the top_k most frequent unigrams in the corpus.
    '''
    unigram_counts = Counter(word for sent in corpus for word in sent)
    top_k_unigrams = [word for word, _ in unigram_counts.most_common(top_k)]
    return set(top_k_unigrams)


def restrict_vocab(corpus: List[List[str]], vocab: Set[str]) -> List[List[str]]:
    '''
    Make the corpus fit inside the vocabulary by replacing out-of-vocabulary words with <unk>.
    '''
    new_corpus = []
    for i, sent in enumerate(corpus):
        new_corpus.append([word if word in vocab else '<unk>' for word in sent])
    return new_corpus



def train_test_split(
    corpus: List[List[str]],
    train_split: float = 0.7,
    num_folds: int = None,
    fold: int = None
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Normally, splits corpus at train_split ratio. Default is 0.7, i.e. 70% train, 30% test.
    However, if num_folds and fold are given, returns the k-th cross-validation fold.

    Hint: Reuse this function for Ex. 6.1.2.
    """
    if num_folds is not None and fold is not None:
        fold_size = len(corpus) // num_folds
        start = fold * fold_size #start of test set
        end = start + fold_size #end of test set
        test_sents = corpus[start:end] #test set is the k-th fold
        train_sents = corpus[:start] + corpus[end:] #train set is the rest of the corpus
    else: #when no folds are given, do a normal train-test split
        split_index = int(len(corpus) * train_split) 
        train_sents = corpus[:split_index]
        test_sents = corpus[split_index:]
    return train_sents, test_sents

class InterpolatedModel:
    def __init__(
        self,
        train_sents: List[List[str]],
        test_sents: List[List[str]],
        alpha: float = 0,
        order: int = 2
    ):
        '''
        Initializes the InterpolatedModel.
        Args:
            train_sents: List of sentences from the training section of the corpus.
            test_sents: List of sentences from the testing section of the corpus.
            alpha: Smoothing factor for Laplace smoothing. Defaults to 0.
            order: The order of n-grams to be used in the model. Defaults to 2.    
        '''
        self.alpha = alpha
        self.order = order
        self.interpolation_weight = 1 / self.order
        
        # Compute n-gram counts for all orders for the training sentences
        self.train_counts = [Counter() for _ in range(self.order + 1)]

        # 0th order counts = corpus size
        total_words = sum(len(sent) for sent in train_sents)
        self.train_counts[0] = {(): total_words}

        for sentence in train_sents:
            for ord_ in range(1, self.order + 1): # For > 0 order n-grams
                self.train_counts[ord_].update(self._get_n_grams(sentence, ord_))

        # Validate that sum of unigram counts = total number of words in the corpus
        assert sum(self.train_counts[1].values()) == total_words, 'Not all unigrams accounted.'

        # Getting vocabulary size for laplace smoothing
        self.vocab_size = len(self.train_counts[1])

        # Getting highest order ngrams from the test set
        self.test_ngrams = [self._get_n_grams(sent, self.order) for sent in test_sents]


    def _get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        '''
        Extracts n-grams from a list of tokens for an arbitrary n (order).
        '''
        n_grams = []
        for i in range(len(tokens) - n + 1):
            n_gram = tuple(tokens[i: i + n])
            n_grams.append(n_gram)
        return n_grams


    def laplace_prob(self, ngram: Tuple[str, ...]) -> float:
        '''
        Calculates the probability of an n-gram using Laplace smoothing.
        '''
        # find the length of the ngram so that can access train_coutns
        order = len(ngram)

        #get the ngram counts
        ngram_count = self.train_counts[order][ngram]

        history = ngram[:-1] #history is the n-1 gram
        history_count = self.train_counts[order - 1][history]

        #laplace smoothing formula:
        prob = (ngram_count + self.alpha) / (history_count + self.alpha * self.vocab_size)
        return prob


    def interpolated_prob(self, ngram: Tuple[str, ...]) -> float:
        '''
        Calculates the interpolated probability of a given n-gram using the Laplace smoothed probabilities.
        Hint: When backing off to a lower order n-gram, cut off the HISTORY by one word.
        '''
        total_prob = 0.0

        #iterate over all  the oders form highest to lowest 
        # toback off to lower orders cutting off the history by one word for each lower order as in the hint given in the question
        # ngram ---> 0 decrement by 1 so sub_ngram = ngram[-order:] will give us the n-gram for the current order and as we decrement the order, it will cut off the history by one word
        for order in range(len(ngram), 0, -1): #start with the highest order and backoff to lower orders
            sub_ngram = ngram[-order:] #cut off the history by one word for each lower order
            prob = self.laplace_prob(sub_ngram)
            total_prob += self.interpolation_weight * prob #use the same interpolation weight for all orders as given in the question

        return total_prob 


    def perplexity(self) -> float:
        '''
        Calculates the perplexity of the language model for test n-grams using the interpolated probabilities.
        Hint: Perplexity should be calculated across all n-grams in the test set, not sentence-wise.
        '''
        #flattern the list of test n-grams to get a single list of n-grams for all sentences in the test set
        all_test_ngrams = [ngram for sent_ngrams in self.test_ngrams for ngram in sent_ngrams]
        N = len(all_test_ngrams) #total number of n-grams in the test set
        log_prob_sum = 0.0
        for ngram in all_test_ngrams:
            prob = self.interpolated_prob(ngram)
            log_prob_sum += math.log2(prob)

        avg_log_prob = log_prob_sum / N
        perplexity = 2**(-avg_log_prob)
        return perplexity
       
