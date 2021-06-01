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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    initial_prob, emission_prob, transition_prob, tag_prob = training(train)
    emission_prob = laplace_smoothening(emission_prob, tag_prob)
    transition_prob = laplace_smoothening(transition_prob, tag_prob)
    predictions = build_viterbi_predictions(emission_prob, transition_prob, test)
    return predictions

def laplace_smoothening(emission_prob, alpha):

    for key in emission_prob:
        words_in_tag = emission_prob[key]
        total_words_in_tag = sum(words_in_tag.values())

        for word in words_in_tag:
            words_in_tag[word] = math.log((words_in_tag[word] + alpha[key]) / (total_words_in_tag + alpha[key]*(len(words_in_tag) + 1)))

        words_in_tag["unknown"] = math.log((alpha[key]) / (total_words_in_tag + alpha[key]*(len(words_in_tag) + 1)))

    # print(emission_prob)
    return emission_prob


def training(train):

    tag_list = {}
    emission_prob = {}
    transition_prob = {}
    hapax = {}
    tag_prob = {}


    for sentence in train:
        previous_tag = "START"

         # emission probability
        for word, tag in sentence:
            if tag in emission_prob:
                tag_map = emission_prob.get(tag)
                tag_map[word] = tag_map.get(word, 0) + 1
                emission_prob[tag] = tag_map
            else:
                emission_prob[tag] = {word: 1}

            if tag in tag_list:
                tag_list[tag] += 1
            else:
                tag_list[tag] = 1

            # transition probability
            if tag in transition_prob:
                tag_map = transition_prob.get(tag)
                tag_map[previous_tag] = tag_map.get(previous_tag, 0) + 1
                transition_prob[tag] = tag_map
                previous_tag = tag
            else:
                transition_prob[tag] = {previous_tag: 1}
                previous_tag = tag

    for key in emission_prob:
        d = emission_prob[key]

        for k in d:

            if d[k] == 1:

                if key in hapax:
                    hapax[key].append(k)
                else:
                    hapax[key] = [k]

    for key in hapax:
        tag_prob[key] = len(hapax[key])

    total_words_in_hapax = sum(tag_prob.values())

    for key in hapax:
        tag_prob[key] /= (total_words_in_hapax * 100000)

    list_of_tags = list(emission_prob.keys())

    for key in list_of_tags:

        if key not in tag_prob:
            tag_prob[key] = 0.00001/total_words_in_hapax


    return 1, emission_prob, transition_prob, tag_prob


def build_viterbi_predictions(emission_prob, transition_prob, test):

    result = []

    for sentence in test:
        tagged_sentence = build_viterbi(sentence, emission_prob, transition_prob)
        result.append(tagged_sentence)

    # print(result)
    return result


def build_viterbi(sentence, emission_prob, transition_prob):

    V = [{}]
    list_of_tags = list(emission_prob.keys())
    number_of_words = len(sentence)

    for tag in list_of_tags:

        if sentence[0] in emission_prob:
            V[0][tag] = {"probability": emission_prob[tag].get(sentence[0]), "previous": None}
        else:
            V[0][tag] = {"probability": emission_prob[tag].get("unknown"), "previous": None}

    for t in range(1, number_of_words):
        V.append({})

        for tag in list_of_tags:
            max_tr_prob = V[t - 1][list_of_tags[0]].get("probability")

            if list_of_tags[0] in transition_prob[tag]:
                max_tr_prob += transition_prob[tag].get(list_of_tags[0])
            else:
                max_tr_prob += transition_prob[tag].get("unknown")

            prev_st_selected = list_of_tags[0]

            for prev_st in list_of_tags[1:]:

                if V[t - 1][prev_st].get("probability"):
                    tr_prob = V[t - 1][prev_st].get("probability")
                else:
                    tr_prob = 0

                if prev_st in transition_prob[tag]:
                    tr_prob += transition_prob[tag].get(prev_st)
                else:
                    tr_prob += transition_prob[tag].get("unknown")

                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob

            if sentence[t] in emission_prob[tag]:
                max_prob += emission_prob[tag].get(sentence[t])
            else:
                max_prob += emission_prob[tag].get("unknown")

            V[t][tag] = {"probability": max_prob, "previous": prev_st_selected}

    opt = []
    max_prob = -10000000
    previous = "None"

    for st, data in V[-1].items():

        if data["probability"] > max_prob:
            max_prob = data["probability"]
            best_st = st

    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous].get("previous"))
        previous = V[t + 1][previous].get("previous")

    l = []

    for index, word in enumerate(sentence):
        l.append((word, opt[index]))

    return l