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
    p_initial, p_emission, p_transition, prob_of_tag = training(train)
    p_emission = laplace_smoothening(p_emission, prob_of_tag)
    p_transition = laplace_smoothening(p_transition, prob_of_tag)
    # print(p_transition)
    predictions = build_viterbi_predictions(p_emission, p_transition, test)
    return predictions


def laplace_smoothening(p_emission, alpha):
    for key in p_emission:
        words_in_tag = p_emission[key]
        total_words_in_tag = sum(words_in_tag.values())
        for word in words_in_tag:
            words_in_tag[word] = math.log((words_in_tag[word] + alpha[key]) / (total_words_in_tag + alpha[key]*(len(words_in_tag) + 1)))
        words_in_tag["UNK"] = math.log((alpha[key]) / (total_words_in_tag + alpha[key]*(len(words_in_tag) + 1)))
    # print(p_emission)
    return p_emission


def training(train):

    # transition probability
    # emission probability
    # initial probability

    p_emission = {}
    p_transition = {}
    p_initial = 1
    tag_list = {}
    hapax = {}

    for line in train[:]:
        previous_tag = "START"
        # emission probability
        for word, tag in line:
            if tag in p_emission:
                each_tag_map = p_emission.get(tag)
                each_tag_map[word] = each_tag_map.get(word, 0) + 1
                p_emission[tag] = each_tag_map
            else:
                p_emission[tag] = {word: 1}

            if tag in tag_list:
                tag_list[tag] += 1
            else:
                tag_list[tag] = 1

            # transition probability
            if tag in p_transition:
                each_tag_map = p_transition.get(tag)
                each_tag_map[previous_tag] = each_tag_map.get(previous_tag, 0) + 1
                p_transition[tag] = each_tag_map
                previous_tag = tag
            else:
                p_transition[tag] = {previous_tag: 1}
                previous_tag = tag

    for key in p_emission:
        d = p_emission[key]
        for k in d:
            if d[k] == 1:
                if key in hapax:
                    hapax[key].append(k)
                else:
                    hapax[key] = [k]

    prob_of_tag = {}
    for key in hapax:
        prob_of_tag[key] = len(hapax[key])
    total_words_in_hapax = sum(prob_of_tag.values())
    for key in hapax:
        prob_of_tag[key] /= (total_words_in_hapax * 100000)

    list_of_tags = list(p_emission.keys())

    for key in list_of_tags:
        if key not in prob_of_tag:
            prob_of_tag[key] = 0.00001/total_words_in_hapax


    return p_initial, p_emission, p_transition, prob_of_tag


def build_viterbi_predictions(p_emission, p_transition, test):
    result = []
    i = 0
    for each_line in test[:]:
        # print(i)
        i += 1
        tagged_line = build_viterbi(each_line, p_emission, p_transition)
        result.append(tagged_line)
    # print(result)
    return result


def build_viterbi(line, p_emission, p_transition):
    V = [{}]
    list_of_tags = list(p_emission.keys())
    no_of_words = len(line)

    for tag in list_of_tags:
        if line[0] in p_emission:
            V[0][tag] = {"prob": p_emission[tag].get(line[0]), "prev": None}
        else:
            V[0][tag] = {"prob": p_emission[tag].get('UNK'), "prev": None}

    for t in range(1, no_of_words):
        V.append({})
        for tag in list_of_tags:
            max_tr_prob = V[t-1][list_of_tags[0]].get("prob")

            if list_of_tags[0] in p_transition[tag]:
                max_tr_prob += p_transition[tag].get(list_of_tags[0])
            else:
                max_tr_prob += p_transition[tag].get("UNK")

            prev_st_selected = list_of_tags[0]
            for prev_st in list_of_tags[1:]:
                if V[t-1][prev_st].get("prob"):
                    tr_prob = V[t-1][prev_st].get("prob")
                else:
                    tr_prob = 0

                if prev_st in p_transition[tag]:
                    tr_prob += p_transition[tag].get(prev_st)
                else:
                    tr_prob += p_transition[tag].get("UNK")

                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob

            if line[t] in p_emission[tag]:
                max_prob += p_emission[tag].get(line[t])
            else:
                max_prob += p_emission[tag].get("UNK")
            V[t][tag] = {"prob": max_prob, "prev": prev_st_selected}

    opt = []
    max_prob = -9999999
    previous = 'None'

    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t+1][previous].get("prev"))
        previous = V[t+1][previous].get("prev")
    opt.pop(0)
    opt.insert(0, 'START')

    l = []
    for index, word in enumerate(line):
        l.append((word, opt[index]))
    return l