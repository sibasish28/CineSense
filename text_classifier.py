import json
from typing import Optional

import numpy as np
import math
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('punkt_tab')



# stemmer
stemmer = PorterStemmer()


def loadData(fileName):
    # Reading the data and removing columns that are not important.
    print(f'loading dataset "{fileName}"...')
    return pd.read_csv(fileName, sep=',', encoding='latin-1',
                       usecols=lambda col: col not in ["Unnamed: 2",
                                                       "Unnamed: 3",
                                                       "Unnamed: 4"])


def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review.
    '''

    cleaned_review = review

    # remove links
    cleaned_review = re.sub(r"<br />", "", cleaned_review)

    # punctuation
    cleaned_review = re.sub(f"[{string.punctuation}]", "", cleaned_review)

    # lowercase
    cleaned_review = cleaned_review.lower()

    # tokenize
    cleaned_review = word_tokenize(cleaned_review)

    # stopwords
    cleaned_review = [word for word in cleaned_review if
                      word not in stopwords.words('english')]

    # stemming
    cleaned_review = [stemmer.stem(token) for token in cleaned_review]

    # recombine tokens
    cleaned_review = " ".join(cleaned_review)

    return cleaned_review


def find_occurrence(frequency, word, label):
    '''
    Params:
        frequency: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word

        { (word,label):num_occurrence }
    Return:
        n: the number of times the word with its corresponding label appears.
    '''

    n = frequency[(word, label)] if (word, label) in frequency else 0

    return n


def load_model(model_file):
    model_parameters = {}

    # Save the model parameters to a file
    with open(model_file, 'r') as file:
        model_file = json.load(file)

        model_parameters['logprior'] = model_file.get('logprior')
        model_parameters['loglikelihood'] = model_file.get('loglikelihood')

    return model_parameters


def main():
    df = loadData('movie_reviews.csv')

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(df['review'],
                                                                df['sentiment'],
                                                                train_size=1000)

    # map to numerical data
    output_map = {'positive': 0, 'negative': 1}
    reverse_map = {0: 'positive', 1: 'negative'}
    y_train_a = y_train_a.map(output_map)

    freqs_a = review_counter({}, X_train_a, y_train_a)
    print('training model...')
    logprior, loglikelihood = train_naive_bayes(freqs_a, X_train_a, y_train_a)
    print('done!\n\n\n')

    while True:
        prompt_input = input(
            "\n\nEnter a movie review or enter 'X' to exit...\n")
        if prompt_input.lower() == 'x':
            break

        prediction = naive_bayes_predict(prompt_input, logprior, loglikelihood)
        print(f'Your review is: {reverse_map[prediction]}')
    print("Exiting the program.")


def review_counter(output_occurrence, reviews, positive_or_negative):
    '''
    Params:
        output_occurrence: a dictionary that will be used to map each pair to its frequency
        reviews: a list of reviews
        positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1)
    Return:
        output: a dictionary mapping each pair to its frequency
    '''
    ## Steps :
    # define the key, which is the word and label tuple
    # if the key exists in the dictionary, increment the count
    # else, if the key is new, add it to the dictionary and set the count to 1

    for label, review in zip(positive_or_negative, reviews):
        split_review = clean_review(review).split()
        for word in split_review:
            key = (word, label)
            output_occurrence[key] = output_occurrence[
                                         key] + 1 if key in output_occurrence else 1
    return output_occurrence


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of reviews
        train_y: a list of labels corresponding to the reviews (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''

    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([word for word, label in freqs.keys()])
    V = len(vocab)

    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in freqs.keys():
        # if the label is positive (equals zero)
        if pair[1] == 0:
            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += freqs[pair]

        # else, the label is negative
        else:
            # Increment the number of negative words by the count for this (word,label) pair
            num_neg += freqs[pair]

    # Calculate num_doc, the number of documents
    num_doc = len(train_x)

    # Calculate D_pos, the number of positive documents
    pos_num_docs = (train_y == 1).sum() + 1

    # Calculate D_neg, the number of negative documents
    neg_num_docs = (train_y == 0).sum() + 1

    # Calculate logprior
    logp = {1: np.log(pos_num_docs / num_doc),
            0: np.log(neg_num_docs / num_doc)}
    # scalar log prior since binary class
    logprior = np.log(pos_num_docs / neg_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = (freqs[(word, 1)] if (word, 1) in freqs else 0) + 1
        freq_neg = (freqs[(word, 0)] if (word, 0) in freqs else 0) + 1
        freq_total = freq_pos + freq_neg

        # calculate the probability that each word is positive, and negative
        p_w_pos = freq_pos / freq_total
        p_w_neg = freq_neg / freq_total

        # calculate the log likelihood of the word
        loglikelihood[word] = [np.log(p_w_pos), np.log(p_w_neg)]

    return logprior, loglikelihood


def naive_bayes_predict(review, logprior, loglikelihood, categoricalLabel: Optional[bool] = False):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''

    # process the review to get a list of words
    word_l = clean_review(review).split()

    # probability for each word
    word_p = {word: 0 for word in word_l}

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob = logprior
    prob_pos = logprior
    prob_neg = logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            # total_prob += None
            prob_pos += loglikelihood[word][0]
            prob_neg += loglikelihood[word][1]

            # save the log likelihood
            word_p[word] = loglikelihood[word]

    # print(
    #     'each token and their log-likelihoods for each class [positive, negative]')
    # for key, value in word_p.items():
    #     print(f'tokenized word: "{key}", likelihood: "{value}"')

    classification = 0 if prob_pos < prob_neg else 1

    if categoricalLabel:
        # map from numerical data
        reverse_map = {0: 'positive', 1: 'negative'}
        return reverse_map[classification]

    return classification


# if __name__ == '__main__':
#     main()
