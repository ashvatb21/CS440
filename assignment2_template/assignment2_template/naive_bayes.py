# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

"""

# Acknowledgment - Arpandeep Khatua

import numpy as numpy
import math



def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set

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
        ham_dict[word] = math.log10((ham_dict[word] + smoothing_parameter)/(ham_total + smoothing_parameter * (len(ham_dict) + 1)))

    for word in spam_dict:
        spam_dict[word] = math.log10((spam_dict[word] + smoothing_parameter)/(spam_total + smoothing_parameter * (len(spam_dict) + 1)))

    ham_dict["unknown"] = math.log10(smoothing_parameter/(ham_total + smoothing_parameter * (len(ham_dict) + 1)))
    spam_dict["unknown"] = math.log10(smoothing_parameter / (spam_total + smoothing_parameter * (len(spam_dict) + 1)))


    # Development Phase
    prediction = []

    for wordList in dev_set:
        ham_prob = math.log10(pos_prior)
        spam_prob = math.log10(1 - pos_prior)

        for word in wordList:

            if word in ham_dict:
                ham_prob += ham_dict[word]
            else:
                ham_prob += ham_dict["unknown"]

            if word in spam_dict:
                spam_prob += spam_dict[word]
            else:
                spam_prob += spam_dict["unknown"]

        if ham_prob < spam_prob:
            prediction.append(0)

        else:
            prediction.append(1)

    return prediction
    