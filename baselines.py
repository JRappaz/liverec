import torch
import numpy as np
from collections import defaultdict,Counter
from sampling import *

class POP():
    def __init__(self, args, data_tr):
        self.args = args
        self.cnt = defaultdict(int,dict(Counter(data_tr['streamer'].tolist())))
    def eval(self): pass
    def compute_rank(self,data,store,k=10):
        inputs = data[:,:,3] # inputs 
        pos    = data[:,:,5] # targets
        xtsy   = data[:,:,6] # targets ts
 
        mask = torch.ones_like(pos[:,-1]).type(torch.bool)
        for b in range(pos.shape[0]):
            avt = pos[b,:-1]
            avt = avt[avt!=0]
            mask[b] = pos[b,-1] in avt
            store['ratio'] += [float(pos[b,-1] in avt)]
            
        for b in range(inputs.shape[0]):
            step = xtsy[b,-1].item()
            av = self.args.ts[step]
            
            scores = np.array([self.cnt[a] for a in av])
            iseq = pos[b,-1] == torch.LongTensor(av).to(self.args.device)
            idx = torch.where(iseq)
            idx = int(idx[0].item())
            rank = np.where(scores.argsort()[::-1]==idx)

            if mask[b]: # rep
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        return store

class REP():
    def __init__(self, args, data_tr):
        self.args = args
    def eval(self): pass
    def compute_rank(self,data,store,k=10):
        inputs = data[:,:,3] # inputs 
        pos    = data[:,:,5] # targets
        xtsy   = data[:,:,6] # targets ts
        
        mask = torch.ones_like(pos[:,-1]).type(torch.bool)
        for b in range(pos.shape[0]):
            avt = pos[b,:-1]
            avt = avt[avt!=0]
            mask[b] = pos[b,-1] in avt
            store['ratio'] += [float(pos[b,-1] in avt)]
            
        for b in range(inputs.shape[0]):
            step = xtsy[b,-1].item()
            av = self.args.ts[step]
            
            cnt = defaultdict(int,dict(Counter(inputs[b,:-1].tolist())))
            scores = np.array([cnt[a] for a in av])
            iseq = pos[b,-1] == torch.LongTensor(av).to(self.args.device)
            idx = torch.where(iseq)
            idx = int(idx[0].item())
            rank = np.where(scores.argsort()[::-1]==idx)

            if mask[b]: # rep
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        return store

class MF(torch.nn.Module):
    def __init__(self, args):
        super(MF, self).__init__()
        self.args = args
        self.item_num = args.N
        self.mf_avg = False

        self.item_bias = torch.nn.Embedding(args.N+1, 1, padding_idx=0)
        self.item_embedding = torch.nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.user_embedding = torch.nn.Embedding(args.M+1, args.K, padding_idx=0)

    def forward(self,users,items):
        ui = self.user_embedding(users)
        ii = self.item_embedding(items)
        ib = self.item_bias(items).squeeze()
        return (ui * ii).sum(-1) + ib

    def train_step(self, data, use_ctx=False): # for training
        bs     = data.shape[0]
        inputs = data[:,:,3] 
        pos    = data[:,:,5] 
        users  = data[:,:,4] 
        neg    = sample_negs(data,self.args).to(self.args.device)

        pos_logits = self(users,pos)
        neg_logits = self(users,neg) 

        loss = - (pos_logits - neg_logits).sigmoid().log()
        loss = loss[inputs!=0].sum()

        return loss

    def compute_rank(self,data,store,k=10):
        inputs = data[:,:,3] 
        pos    = data[:,:,5] 
        users  = data[:,:,4] 
        xtsy   = data[:,:,6] # targets ts

        mask = torch.ones_like(pos[:,-1]).type(torch.bool)
        for b in range(pos.shape[0]):
            avt = pos[b,:-1]
            avt = avt[avt!=0]
            mask[b] = pos[b,-1] in avt
            store['ratio'] += [float(pos[b,-1] in avt)]
            
        for b in range(inputs.shape[0]):
            step = xtsy[b,-1].item()
            av = torch.LongTensor(self.args.ts[step]).to(self.args.device)
            av_embs = self.item_embedding(av)
 
            if self.mf_avg:
                inp = inputs[b,:]
                mean_vec = self.item_embedding(inp[inp!=0])
                scores  = (mean_vec.mean(0).unsqueeze(0) * av_embs).sum(-1) 
                scores += self.item_bias(av).squeeze()
            else:
                u_vec = self.user_embedding(users[b,-1])
                scores  = (u_vec.unsqueeze(0) * av_embs).sum(-1) 
                scores += self.item_bias(av).squeeze()

            iseq = pos[b,-1] == av
            idx = torch.where(iseq)
            idx = idx[0] 
            rank = torch.where(torch.argsort(scores, descending=True)==idx)[0].item()

            if mask[b]: # rep
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        return store

class FPMC(torch.nn.Module):
    def __init__(self, args):
        super(FPMC, self).__init__()
        self.args = args
        self.item_num = args.N

        self.user_embs = torch.nn.Embedding(args.M+1, args.K, padding_idx=0)
        self.item_embs = torch.nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.prev_embs = torch.nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.next_embs = torch.nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.item_bias = torch.nn.Embedding(args.N+1, 1, padding_idx=0)

    def forward(self,users,prev,items):
        ui = self.user_embs(users)
        ii = self.item_embs(items)

        ip = self.prev_embs(prev)
        ic = self.next_embs(items)
        
        ib = self.item_bias(items).squeeze()
        return (ui * ii).sum(-1) + (ip * ic).sum(-1) + ib 

    def train_step(self, data, use_ctx=False): # for training
        bs     = data.shape[0]
        inputs = data[:,:,3] 
        pos    = data[:,:,5] 
        users  = data[:,:,4] 
        neg    = sample_negs(data,self.args).to(self.args.device)

        pos_logits = self(users,inputs,pos)
        neg_logits = self(users,inputs,neg) 

        loss = - (pos_logits[inputs!=0] - neg_logits[inputs!=0]).sigmoid().log()
        return loss.sum()

    def compute_rank(self,data,store,k=10):
        inputs = data[:,:,3] 
        pos    = data[:,:,5] 
        users  = data[:,:,4] 
        xtsy   = data[:,:,6] # targets ts
        neg    = sample_negs(data,self.args).to(self.args.device)

        mask = torch.ones_like(pos[:,-1]).type(torch.bool)
        for b in range(pos.shape[0]):
            avt = pos[b,:-1]
            avt = avt[avt!=0]
            mask[b] = pos[b,-1] in avt
            store['ratio'] += [float(pos[b,-1] in avt)]
            
        for b in range(inputs.shape[0]):
            step = xtsy[b,-1].item()
            av = torch.LongTensor(self.args.ts[step]).to(self.args.device)

            ui = self.user_embs(users[b,-1]) 
            pi = self.prev_embs(inputs[b,-1])
                   
            scores  = (ui.unsqueeze(0) * self.item_embs(av)).sum(-1) 
            scores += (pi.unsqueeze(0) * self.next_embs(av)).sum(-1) 
            scores += self.item_bias(av).squeeze()

            iseq = pos[b,-1] == av
            idx = torch.where(iseq)
            idx = idx[0] 
            rank = torch.where(torch.argsort(scores, descending=True)==idx)[0].item()

            if mask[b]: # rep
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        return store

