import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

#%%
class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=15,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

#%%
class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


#%%
class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores

        N = len(self.X)
        
        best_score = np.float('inf')
        for n_states in range(self.min_n_components,self.max_n_components + 1):
            score = np.float('inf')
            p = (n_states**2) + 2*len(self.X[0])*n_states
            try:
                hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                score = hmm_model.score(self.X,self.lengths)
                score = -2*score + p*np.log(N)
            except:
                hmm_model = None
            if score < best_score:
                model = hmm_model
                best_score = score
        if hmm_model == None:
            try:
                model = hmm_model = GaussianHMM(n_components=self.n_constant, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            except:
                return None
        return model


#%%
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        best_score = -np.float('inf')
        M = len(self.X) # number of words
        for n_states in range(self.min_n_components, self.max_n_components + 1):
            score = -np.float('inf')
            try:
                hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                score_word = hmm_model.score(self.X,self.lengths)
                score_other = 0;
                # Get score for all the other words
                for key,value in self.hwords.items():
                    currentX, currentLength = value
                    score_other += hmm_model.score(currentX, currentLength)
                # Subtract the score word to get the correct calculation
                score_other = score_other - score_word 
                score = score_word - score_other/(M-1)
            except:
                hmm_model = None
                score = np.float('inf')
            if score > best_score:
                model = hmm_model
                best_score = score
        if hmm_model == None:
            try:
                model = hmm_model = GaussianHMM(n_components=self.n_constant, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            except:
                return None
        return model


#%%
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        split_method = KFold(n_splits=2) # two splits to get less failures
        best_score = -np.float('inf')
        model = None
        for n_comp in range(self.min_n_components,self.max_n_components + 1):
            # Initialize score and counter for each component
            score = 0
            n = 0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_sequence,train_length = combine_sequences(cv_train_idx,self.sequences)
                    test_sequence,test_length = combine_sequences(cv_test_idx,self.sequences)
                    hmm_model = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_sequence, train_length)
                    score += hmm_model.score(test_sequence, test_length)
                    n += 1
                score = score/n
            except:
                # Return none if there is no predicted model at all, otherwise continue
                hmm_model = None
                score = -np.float('inf')
     
            if score > best_score:
                model = hmm_model
                best_score = score        
        return model