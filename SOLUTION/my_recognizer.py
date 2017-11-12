import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    # Iterate over every word in the test set and extract the probability and best guess word
     
    for ix,test_word in enumerate(test_set.wordlist):
        # dictionary for probabibily of each model to the current word of the test set
        probs_word = dict()
        X, lengths = test_set.get_item_Xlengths(ix)
        # Best guess score
        best_score = -np.float('inf')
        guess_word = None

        # iterate over models to get the score of the test_word
        for word, model in models.items():
            try:
                score = model.score(X,lengths)
                if score > best_score:
                    guess_word = word
                    best_score = score
            except:
                score = -np.float('inf')
            probs_word[word] = score
        guesses.append(guess_word)
        probabilities.append(probs_word)       
    return probabilities,guesses