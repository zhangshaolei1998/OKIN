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
os.environ['CUDA_VISIBLE_DEVICES']='2'

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
        # print(x.shape)

        a = x.unsqueeze(2).float()
        # print(a.shape)
        atten_a = torch.bmm(a, a.transpose(1, 2))
        # print(atten_a.shape)
        atten_a = np.float64(atten_a.cpu() > 0)
        atten_a = torch.Tensor(atten_a).cpu()
        # print(atten_a)

        atten_a = atten_a.cpu().numpy()

        print(atten_a.shape[0])

        for i in range(0,atten_a.shape[0]):

            atten_a[i] = atten_a[i] - np.diag(np.diag(atten_a[i]))

        atten_a = np.where(atten_a > 0, atten_a, -999999999)
        atten_a = torch.Tensor(atten_a).cpu().type(torch.FloatTensor)


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
                score=self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit=x_logit.transpose(2, 0)
                score=torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score=-self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit=torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len_opi, batch_first=True)
                score=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score

def label_rest_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("Opinions")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ':
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("Opinion")
            opin.attrib['target']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)

def label_laptop_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("aspectTerms")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ' or ord(c)==160:
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("aspectTerm")
            opin.attrib['term']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)  
    

def test(model,test_X, raw_X, domain, command, template, batch_size=128, crf=False):
    pred_y=np.zeros((test_X.shape[0], 83), np.int16)
    atten_weight = np.zeros((test_X.shape[0], 83,83), np.float64)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len=np.sum(test_X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_test_X_len.argsort()[::-1]
        batch_test_X_len=batch_test_X_len[batch_idx]
        batch_test_X_mask=(test_X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_test_X=test_X[offset:offset+batch_size][batch_idx]
        #print("*****test:%d", offset)
        #print(test_X[offset])
        #print("*****:%d", offset)
        #print(batch_test_X[offset])
        batch_test_X_mask=torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().cuda() )
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().cuda() )
        #print("*****:%d",offset)
        #print(batch_test_X[offset])
        #batch_pred_y, batch_atten_weight = model(batch_test_X, batch_test_X_len, batch_test_X_len, batch_test_X_mask,testing=True)
        batch_pred_y= model(batch_test_X, batch_test_X_len, batch_test_X_len, batch_test_X_mask,
                                                 testing=True)
        #print("batch_pred_y维度:", type(batch_pred_y))
        #print("batch_atten_weight维度:", batch_atten_weight[1].shape)
        r_idx=batch_idx.argsort()
        if crf:
            batch_pred_y=[batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y) ):
                for jx in range(len(batch_pred_y[ix]) ):
                    pred_y[offset+ix,jx]=batch_pred_y[ix][jx]
        else:
            batch_pred_y=batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            #atten_weight[offset:offset + batch_size,:batch_atten_weight.shape[1],:batch_atten_weight.shape[2]] = batch_atten_weight.data.cpu().numpy()[r_idx]
            pred_y[offset:offset+batch_size,:batch_pred_y.shape[1]]=batch_pred_y
            #print("*****:%d",offset)
            #print(atten_weight[offset])

    model.train()
    '''
    for i in range(0,20):
        #print("test:",test_X[i])
        print("test %d:" % (i), file=of)
        print(test_X[i], file=of)
        #print("pres:", pred_y[i])
        print("pres %d:" % (i), file=of)
        print(pred_y[i], file=of)
        #print("atten:",atten_weight[i])
        print("atten %d:" % (i), file=of)
        print(atten_weight[i], file=of)
    '''
    print("*****************")
    #np.savez('attention.npz', test_X=test_X, pred_y=pred_y, atten=atten_weight)
    print("+++++++++++++++++")
    assert len(pred_y)==len(test_X)

    np.savez(domain+'_pred.npz', test_X=test_X, pred_y=pred_y)
    
    command=command.split()

    #print(command)
    if domain=='restaurant':
        label_rest_xml(template, command[6], raw_X, pred_y)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[9][10:])
    elif domain=='laptop':
        label_laptop_xml(template, command[4], raw_X, pred_y)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[15])

def evaluate(runs, data_dir, model_dir, domain, command, template):
    ae_data=np.load(data_dir+domain+".npz")
    with open(data_dir+domain+"_raw_test.json") as f:
        raw_X=json.load(f)
    results=[]
    for r in range(runs):
        model=torch.load(model_dir+domain+str(r) )
        result=test(model, ae_data['test_X'], raw_X, domain, command, template, crf=False)
        results.append(result)
    print(sum(results)/len(results) )

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--domain', type=str, default="laptop")


    args = parser.parse_args()
    of = open('attention.txt', 'w')
    if args.domain=='restaurant':
        command="java -cp script/A.jar absa16.Do Eval -prd data/official_data/pred.xml -gld data/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template="data/official_data/EN_REST_SB1_TEST.xml.A"
    elif args.domain=='laptop':
        command="java -cp script/eval.jar Main.Aspects data/official_data/pred.xml data/official_data/Laptops_Test_Gold.xml"
        template="data/official_data/Laptops_Test_Data_PhaseA.xml"

    evaluate(args.runs, args.data_dir, args.model_dir,args.domain, command, template)
