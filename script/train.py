import argparse
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
import math
import random
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def batch_generator(X, y, y_opi, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len=np.sum(X[offset:offset+batch_size]!=0, axis=1)
        #print(batch_X_len)
        #print(batch_X_len.shape)
        batch_X_len_opi = np.sum(X[offset:offset + batch_size] !=0, axis=1)
        #print(batch_X_len_opi)
        #print(batch_X_len_opi.shape)
        batch_idx=batch_X_len.argsort()[::-1]
        batch_X_len=batch_X_len[batch_idx]
        batch_X_len_opi = batch_X_len_opi[batch_idx]
        batch_X_mask=(X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_X=X[offset:offset+batch_size][batch_idx]
        batch_y=y[offset:offset+batch_size][batch_idx]
        batch_y_opi = y_opi[offset:offset + batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda() )
        batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda() )
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda() )
        batch_y_opi = torch.autograd.Variable(torch.from_numpy(batch_y_opi).long().cuda())
        if len(batch_y.size() )==2 and not crf:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if len(batch_y_opi.size() )==2 and not crf:
            batch_y_opi=torch.nn.utils.rnn.pack_padded_sequence(batch_y_opi, batch_X_len_opi, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_y_opi, batch_X_len,batch_X_len_opi, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_y_opi, batch_X_len,batch_X_len_opi, batch_X_mask)

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

    def forward(self, x, x_len,x_len_opi, x_mask, x_tag=None, x_tag_opi=None, testing=False):
        #print(x.shape)

        a=x.unsqueeze(2).float()
        #print(a.shape)
        atten_a=torch.bmm(a,a.transpose(1,2))
        #print(atten_a.shape)
        atten_a=np.float64(atten_a.cpu() > 0)
        atten_a=torch.Tensor(atten_a).cpu().type(torch.FloatTensor)
        #print(atten_a)

        atten_a=atten_a.cpu().numpy()

        #print(atten_a.shape[0])

        for i in (0,atten_a.shape[0]-1):
            atten_a[i]=atten_a[i]-np.diag(np.diag(atten_a[i]))

        atten_a=np.where(atten_a > 0, atten_a, -999999999)
        atten_a=torch.Tensor(atten_a).cpu().type(torch.FloatTensor)

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
                x_logit = x_logit.transpose(2, 0)
                score = torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score = -self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit = torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score1 = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)

                #print(type(x_logit_opi))
                x_logit_opi = torch.nn.utils.rnn.pack_padded_sequence(x_logit_opi, x_len_opi, batch_first=True)
                #print(type(x_logit_opi))
                score2 = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit_opi.data), x_tag_opi.data)
                # print('aspect_loss:',score1,'opinion_loss:',score2)
        return score1, score2


def valid_loss(model, valid_X, valid_y, valid_y_opi, crf=False):
    model.eval()
    losses1 = []
    losses2 = []
    for batch in batch_generator(valid_X, valid_y, valid_y_opi, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_y_opi, batch_valid_X_len,batch_valid_X_len_opi, batch_valid_X_mask = batch
        loss1, loss2 = model(batch_valid_X, batch_valid_X_len,batch_valid_X_len_opi, batch_valid_X_mask,batch_valid_y, batch_valid_y_opi)
        losses1.append(loss1.item())
        losses2.append(loss2.item())
    model.train()
    return sum(losses1) / len(losses1), sum(losses2) / len(losses2)


def train(train_X, train_y, train_y_opi, valid_X, valid_y, valid_y_opi, model, model_fn,a,way, optimizer, parameters,optimizer2, parameters2, epochs=200, batch_size=128, crf=False):
    best_loss = float("inf")
    best_loss1 = float("inf")

    valid_history = []
    train_history = []
    for epoch in range(epochs):
        print('epoch', epoch, ':', end='')
        print('epoch', epoch, ':', end='', file=of)
        for batch in batch_generator(train_X, train_y, train_y_opi, batch_size, crf=crf):
            # print(batch,' ', end='')
            batch_train_X, batch_train_y, batch_train_y_opi, batch_train_X_len,batch_train_X_len_opi, batch_train_X_mask = batch
            loss1, loss2 = model(batch_train_X, batch_train_X_len,batch_train_X_len_opi, batch_train_X_mask, batch_train_y, batch_train_y_opi)
            loss = loss1 + a * loss2


            optimizer2.zero_grad()
            loss2.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(parameters2, 1.)
            optimizer2.step()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)

            optimizer.step()

        loss1, loss2 = valid_loss(model,train_X, train_y, train_y_opi, crf=crf)
        loss = loss1 + a * loss2
        print("train_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss), end='')
        print("train_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss), end='', file=of)
        train_history.append(loss)
        loss1, loss2 = valid_loss(model, valid_X, valid_y, valid_y_opi, crf=crf)
        loss = loss1 + a * loss2
        print(" || valid_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss))
        print(" || valid_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss), file=of)
        valid_history.append(loss)
        if way==1:
            if loss < best_loss and loss1<best_loss1:
                best_loss = loss
                best_loss1 = loss1
                torch.save(model, model_fn)
                print("update")
                print("update", file=of)
        if way==2:
            if loss < best_loss or loss1<best_loss1:
                best_loss = loss
                best_loss1 = loss1
                torch.save(model, model_fn)
                print("update")
                print("update", file=of)

        if way==3:
            if loss < best_loss:
                best_loss = loss
                best_loss1 = loss1
                torch.save(model, model_fn)
                print("update")
                print("update", file=of)

        if way==4:
            if loss1<best_loss1:
                best_loss = loss
                best_loss1 = loss1
                torch.save(model, model_fn)
                print("update")
                print("update", file=of)
        shuffle_idx = np.random.permutation(len(train_X))
        train_X = train_X[shuffle_idx]
        train_y = train_y[shuffle_idx]
        train_y_opi = train_y_opi[shuffle_idx]
    model = torch.load(model_fn)
    return train_history, valid_history

def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, a,way, batch_size=128):
    gen_emb=np.load(data_dir+"gen.vec.npy")
    domain_emb=np.load(data_dir+domain+"_emb.vec.npy")

    ae_data=np.load(data_dir+domain+"_data.npz")

    #print("gen_emb:", gen_emb.shape)
    #print("domain_emb:", domain_emb.shape)



    valid_X=ae_data['sentences'][-valid_split:]
    valid_y=ae_data['aspect_tags'][-valid_split:]
    valid_y_opi = ae_data['opinion_tags'][-valid_split:]
    train_X=ae_data['sentences'][:-valid_split]
    train_y=ae_data['aspect_tags'][:-valid_split]
    train_y_opi = ae_data['opinion_tags'][:-valid_split]
    '''
    ae_data = np.load(data_dir + domain + ".npz")

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    valid_y_opi = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]
    train_y_opi = ae_data['train_y'][:-valid_split]
    
    print("valid_X:", valid_X.shape)
    print("valid_y:", valid_y.shape)
    print("train_X:", train_X.shape)
    print("train_y:", train_y.shape)

    print("x:")
    for i in range(0,5):
        print(train_X[i])
    print("y:")
    for i in range(0, 5):
        print(train_y[i])
    '''
    for r in range(runs):
        print(r)
        of.write(str(r))
        model=Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters, lr=lr)

        parameters2 = [p for p in model.parameters() if p.requires_grad]
        optimizer2 = torch.optim.Adam(parameters2, lr=lr)
        train_history, valid_history = train(train_X, train_y, train_y_opi, valid_X, valid_y, valid_y_opi, model,
                                             model_dir + domain + str(r),a, way, optimizer, parameters, optimizer2,
                                             parameters2, epochs, crf=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--valid', type=int, default=150) #number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--updateway', type=int, default=1)#1:loss，loss1同时小。2：loss小或loss1小。3：loss小。:4：loss1小
    #parser.add_argument('--asp_layer', type=int, default=4)
    #parser.add_argument('--opi_layer', type=int, default=5)


    args = parser.parse_args()
    of = open('out.txt', 'w')
    run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout,args.a,args.updateway, args.batch_size)
    of.close()
