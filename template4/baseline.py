# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
# Acknowledgment: Arpandeep Khatua
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_class, tag = training(train)
    predicts = testing(test, word_class, tag)
    return predicts
    
def training(train):

    word_class = {}
    most_common_tag = {}

    for sentence in train:

        for word, tag in sentence:

            if word in word_class:
                each_word = word_class.get(word)
                each_word[tag] = each_word.get(tag, 0) + 1
                word_class[word] = each_word
            else:
                word_class[word] = {tag: 1}

            if tag in most_common_tag:
                most_common_tag[tag] += 1
            else:
                most_common_tag[tag] = 1

    for word in word_class:
        word_class[word] = max(word_class[word], key=word_class[word].get)

    return word_class, max(most_common_tag, key=most_common_tag.get)


def testing(test, word_class, tag):
    predicts = []
    for sentence in test:
        liner = []
        for word in sentence:
            if word in word_class:
                liner.append((word, word_class[word]))
            else:
                liner.append((word, tag))
        predicts.append(liner)
    return predicts