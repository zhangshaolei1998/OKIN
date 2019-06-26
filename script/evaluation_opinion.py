# -*- coding: utf-8 -*

import argparse
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
import sys
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output
import subprocess
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.6, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)

        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(dropout)

        self.dropout2 = torch.nn.Dropout(dropout)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae = torch.nn.Linear(512, num_classes)

        self._conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self._conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self._dropout = torch.nn.Dropout(dropout)

        self._fc1 = torch.nn.Linear(256, 256, bias=False)
        self._fc2 = torch.nn.Linear(256, 256, bias=False)

        self._conv3 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self._conv4 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self._conv5 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self._conv6 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self._conv7 = torch.nn.Conv1d(512, 512, 3, padding=1)

        self._linear_ae = torch.nn.Linear(256, num_classes)
        self.fc3 = torch.nn.Linear(166, 83)
        self.atten1 = torch.nn.Linear(166, 83)
        self.atten = torch.nn.Linear(256, 256, bias=False)
        self.atten_w = torch.nn.Linear(83, 1)

        self.crf_flag = crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(num_classes)

    def forward(self, x, x_len, x_len_opi, x_mask, x_tag=None, x_tag_opi=None, testing=False):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        op_conv = x_emb
        x_emb = self.dropout(x_emb).transpose(1, 2)

        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        as_conv = x_conv

        op_conv = self.dropout2(op_conv).transpose(1, 2)
        op_conv = torch.nn.functional.relu(torch.cat((self._conv1(op_conv), self._conv2(op_conv)), dim=1))
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv3(op_conv))
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv4(op_conv))
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv5(op_conv))
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv6(op_conv))  # [128, 256, 83]

        as_conv = as_conv.transpose(1, 2)  # [128, 83, 256]
        op_conv = op_conv.transpose(1, 2)  # [128, 83, 256]

        x_logit_opi = self._linear_ae(op_conv)
        ''''
        atten = F.relu(self.atten(op_conv))  # [128, 83, 256]
        atten = torch.bmm(as_conv, atten.transpose(1, 2))# [128, 83, 256]

        #atten=torch.mul(atten_a.type(torch.FloatTensor),F.relu(atten).type(torch.FloatTensor)).type(torch.FloatTensor)

        atten_weight = F.softmax(F.relu(atten), dim=1).cuda()
        atten_conv = torch.bmm(atten_weight, op_conv).transpose(1, 2)  # [128, 256, 83]
        ans_conv = torch.cat((as_conv, atten_conv.transpose(1, 2)), dim=2)  # [128, 83, 512]
        '''
        ans_conv = torch.cat((as_conv, op_conv), dim=2)
        x_logit = self.linear_ae(ans_conv)

        if testing:
            if self.crf_flag:
                score = self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit_opi = x_logit_opi.transpose(2, 0)
                score = torch.nn.functional.log_softmax(x_logit_opi).transpose(2, 0)
        else:
            if self.crf_flag:
                score = -self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit = torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score





def test(model, test_X, data_dir, domain,  batch_size=128, crf=False):
    pred_op = np.zeros((test_X.shape[0], 83), np.int16)
    atten_weight = np.zeros((test_X.shape[0], 83, 83), np.float64)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len = np.sum(test_X[offset:offset + batch_size] != 0, axis=1)
        batch_idx = batch_test_X_len.argsort()[::-1]
        batch_test_X_len = batch_test_X_len[batch_idx]
        batch_test_X_mask = (test_X[offset:offset + batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_test_X = test_X[offset:offset + batch_size][batch_idx]
        # print("*****test:%d", offset)
        # print(test_X[offset])
        # print("*****:%d", offset)
        # print(batch_test_X[offset])
        batch_test_X_mask = torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().cuda())
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().cuda())
        # print("*****:%d",offset)
        # print(batch_test_X[offset])
        #batch_pred_y, batch_atten_weight = model(batch_test_X, batch_test_X_len,batch_test_X_len, batch_test_X_mask, testing=True)
        batch_pred_y= model(batch_test_X, batch_test_X_len, batch_test_X_len, batch_test_X_mask,
                                                 testing=True)
        # print("batch_pred_y维度:", type(batch_pred_y))
        # print("batch_atten_weight维度:", batch_atten_weight[1].shape)
        r_idx = batch_idx.argsort()
        if crf:
            batch_pred_y = [batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y)):
                for jx in range(len(batch_pred_y[ix])):
                    pred_op[offset + ix, jx] = batch_pred_y[ix][jx]
        else:
            batch_pred_y = batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            #atten_weight[offset:offset + batch_size, :batch_atten_weight.shape[1], :batch_atten_weight.shape[2]] = \
            #batch_atten_weight.data.cpu().numpy()[r_idx]
            pred_op[offset:offset + batch_size, :batch_pred_y.shape[1]] = batch_pred_y
            # print("*****:%d",offset)
            # print(atten_weight[offset])

    model.train()

    '''
    for i in range(0,5):
        print("test:",test_X[i])

        print("pres:", pred_op[i])
    '''


    assert len(pred_op) == len(test_X)


    preds=0
    golds=0
    common=0

    data=np.array(test_X)
    exist = (data > 0) * 1.0
    factor = np.ones(data.shape[1])
    res = np.dot(exist, factor)
    res=res.astype(int)


    data = np.load(data_dir + domain + "_test.npz")
    gold_op=data['opinion_tags'][:]
    for i in range(0,gold_op.shape[0]):
        for j in range(0,res[i]):
            if gold_op[i][j]==1 :
                golds=golds+1
            if pred_op[i][j]==1 :
                preds=preds+1
            if (gold_op[i][j]==1 ) and gold_op[i][j]==pred_op[i][j] :
                k=j
                flag=1
                while gold_op[i][k-1]!=0 and k<res[i]:
                    if gold_op[i][k]!=pred_op[i][k]:
                        flag=0
                        break

                if(flag):
                    common=common+1
    PRE=common/preds
    REC=common/golds
    F1=2/((1/PRE)+(1/REC))
    print("preds: %f , golds: %f , common: %f | PRE: %f , REC: %f , F1: %f "%(preds,golds,common,PRE,REC,F1) )
    return F1




def evaluate(runs, data_dir, model_dir, domain):
    ae_data = np.load(data_dir + domain + ".npz")

    results = []
    for r in range(runs):
        model = torch.load(model_dir + domain + str(r))
        result = test(model, ae_data['test_X'], data_dir, domain, crf=False)
        results.append(result)
    print(sum(results) / len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--domain', type=str, default="laptop")

    args = parser.parse_args()
    of = open('attention.txt', 'w')
    evaluate(args.runs, args.data_dir, args.model_dir, args.domain)
