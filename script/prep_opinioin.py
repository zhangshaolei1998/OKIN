import numpy as np
from nltk import ngrams
import nltk
import string
import os
import random
import json


def replace(sent):
    nsent = '' + sent[0]
    for i in (1, len(sent) - 1):
        if sent[i] == '-' and sent[i - 1].isalpha() and sent[i + 1].isalpha():
            nsent += " - "
        else:
            nsent += sent[i]

    return nsent


def read_data(path, opi_path):

    # load opinion annotations

    with open('../data/prep_data/word_idx.json') as f:
        word_idx = json.load(f)

    # print(word_idx['keeps'])

    opinions = []
    with open(opi_path) as fp:
        for line in fp:
            opi_record = {}
            items = line.strip().split(', ')
            # print(items)
            for item in items:
                eles = item.split()
                polarity = eles[-1]
                word = ' '.join(eles[:-1])
                opi_record[word] = polarity
            opinions.append(opi_record)
    # print(opinions)
    count=0
    dataset = []
    idx = 0
    j = 0
    fp = open(path, 'r', encoding='utf-8')
    for line in fp:
        record = {}
        flag = 0
        opi_flag = 0
        opi_record = opinions[idx]
        sent, tag_string = line.strip().split("####")
        record['sentence'] = sent
        tag_sequence = tag_string.split(' ')
        words, tags, opi_tags = [], [], []

        asp_record = {}
        for item in tag_sequence:
            # print("********")
            eles = item.split('=')
            # print(eles[0])
            tag = eles[-1]

            asp_tokens = eles[0].replace('-', ' - ')
            asp_tokens = nltk.word_tokenize(asp_tokens)

            for asp_token in asp_tokens:
                asp_record[asp_token] = tag
        # print(idx)
        # print(asp_record)

        sent = sent.replace('-', ' - ')
        # print(sent)
        tokens = nltk.word_tokenize(sent)
        j += 1
        # print(j)
        # print(tokens)
        for token in tokens:
            if token in word_idx:
                words.append(word_idx[token])
            else:
                words.append(0)

            if token in opi_record:
                if opi_flag == 0:
                    opi_tags.append(1)
                    count=count+1
                else:
                    opi_tags.append(2)
                opi_flag = 1
            else:
                opi_tags.append(0)
                opi_flag = 0

            if token in asp_record:
                if asp_record[token] == 'T':
                    if flag == 0:
                        tags.append(1)
                    else:
                        tags.append(2)
                    flag = 1
                else:
                    tags.append(0)
                    flag = 0
            else:
                tags.append(0)
                flag = 0

       
        while len(words) < 83:
            words.append(0)
        while len(tags) < 83:
            tags.append(0)
        while len(opi_tags) < 83:
            opi_tags.append(0)
        record['sent'] = np.array(words)
        # origin aspect tags
        record['aspect_tags'] = np.array(tags)
        record['opinion_tags'] = np.array(opi_tags)

        dataset.append(record)
        idx += 1
   
    sentences, aspect_tags, opinion_tags = [], [], []
    for data in dataset:
        sentences.append(data['sent'])
        aspect_tags.append(data['aspect_tags'])
        opinion_tags.append(data['opinion_tags'])

    for i in range(0, 5):
        print(i)
        print(sentences[i])
        print(opinion_tags[i])


    np.savez('../data/prep_data/restaurant_test.npz', sentences=sentences, aspect_tags=aspect_tags, opinion_tags=opinion_tags)

    print("count:", count)
    print("N opinion:", len(opinion_tags))
    print("N dataset:", len(dataset))
    assert len(opinions) == len(dataset)
    return dataset


if __name__ == "__main__":

    train_path = '../data/prep_data/restaurant_test.txt'

    train_opi_path = '../data/prep_data/restaurant_test_opi.txt'

    train_set = read_data(train_path, train_opi_path)