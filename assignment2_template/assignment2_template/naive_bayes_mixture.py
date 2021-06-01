# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

# Acknowledgment - Arpandeep Khatua

import math

def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set

    # Bigram
    ham_dict_bi = {} 
    spam_dict_bi = {}


    for i in range(len(train_set)):
        wordList = train_set[i]
        label = train_labels[i]
        wordList_bi = []

        for j in range(len(wordList) - 1):
            wordList_bi.append((wordList[j], wordList[j + 1]))

        for word_pair in wordList_bi:

            if label == 1:

                if word_pair not in ham_dict_bi:
                    ham_dict_bi[word_pair] = 1
                else:
                    ham_dict_bi[word_pair] += 1

            else:

                if word_pair not in spam_dict_bi:
                    spam_dict_bi[word_pair] = 1
                else:
                    spam_dict_bi[word_pair] += 1

    ham_total_bi = sum(ham_dict_bi.values())
    spam_total_bi = sum(spam_dict_bi.values())

    for word in ham_dict_bi:
        ham_dict_bi[word] = math.log((ham_dict_bi[word] + bigram_smoothing_parameter) / (ham_total_bi + bigram_smoothing_parameter * (len(ham_dict_bi) + 1)))
    
    for word in spam_dict_bi:
        spam_dict_bi[word] = math.log((spam_dict_bi[word] + bigram_smoothing_parameter) / (spam_total_bi + bigram_smoothing_parameter * (len(spam_dict_bi) + 1)))
    
    ham_dict_bi["unknown"] = math.log(bigram_smoothing_parameter / (ham_total_bi + bigram_smoothing_parameter * (len(ham_dict_bi) + 1)))
    spam_dict_bi["unknown"] = math.log(bigram_smoothing_parameter / (spam_total_bi + bigram_smoothing_parameter * (len(spam_dict_bi) + 1)))
    


    ham_dict = {}
    spam_dict = {}

    for i in range(len(train_set)):
        wordList = train_set[i]
        label = train_labels[i]

        for word in wordList:

            if label == 1:

                if word not in ham_dict:
                    ham_dict[word] = 1
                else:
                    ham_dict[word] += 1

            else:

                if word not in spam_dict:
                    spam_dict[word] = 1
                else:
                    spam_dict[word] += 1


    ham_total = sum(ham_dict.values())
    spam_total = sum(spam_dict.values())

    for word in ham_dict:
        ham_dict[word] = math.log((ham_dict[word] + unigram_smoothing_parameter) / (ham_total + unigram_smoothing_parameter * (len(ham_dict) + 1)))
    
    for word in spam_dict:
        spam_dict[word] = math.log((spam_dict[word] + unigram_smoothing_parameter) / (spam_total + unigram_smoothing_parameter * (len(spam_dict) + 1)))
    
    ham_dict["unknown"] = math.log(unigram_smoothing_parameter / (ham_total + unigram_smoothing_parameter * (len(ham_dict) + 1)))
    spam_dict["unknown"] = math.log(unigram_smoothing_parameter / (spam_total + unigram_smoothing_parameter * (len(spam_dict) + 1)))
    
    # Development Phase
    prediction = []

    for wordList in dev_set:
        ham_prob = math.log(pos_prior)
        spam_prob = math.log(1 - pos_prior)

        for word in wordList:

            if word in ham_dict:
                ham_prob += ham_dict[word]
            else:
                ham_prob += ham_dict["unknown"]

            if word in spam_dict:
                spam_prob += spam_dict[word]
            else:
                spam_prob += spam_dict["unknown"]

        ham_prob_bi = math.log10(pos_prior)
        spam_prob_bi = math.log10(1 - pos_prior)

        wordList_bi = []
        for j in range(len(wordList) - 1):
            wordList_bi.append((wordList[j], wordList[j + 1]))

        for word_pair in wordList_bi:

            if word_pair in ham_dict_bi:
                ham_prob_bi += ham_dict_bi[word_pair]
            else:
                ham_prob_bi += ham_dict_bi["unknown"]

            if word_pair in spam_dict_bi:
                spam_prob_bi += spam_dict_bi[word_pair]
            else:
                spam_prob_bi += spam_dict_bi["unknown"]

        prob_ham = ((1 - bigram_lambda) * ham_prob) + (bigram_lambda * ham_prob_bi)
        prob_spam = ((1 - bigram_lambda) * spam_prob) + (bigram_lambda * spam_prob_bi)

        if prob_ham < prob_spam:
            prediction.append(0)
        else:
            prediction.append(1)

    return prediction