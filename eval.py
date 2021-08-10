import random
from tqdm import tqdm
import torch
from sampling import *
from data import *
import torch.nn.functional as F
import numpy as np

def save_scores(scores, args):
    with open("logs.txt", 'a') as fout:
        fout.write('{};{};{};{};{:.5f};{:.5f};{};{}\n'.format(
                   args.model,
                   args.K,
                   args.fr_ctx,
                   args.fr_rep,
                   args.lr,
                   args.l2,
                   args.seq_len,
                   args.topk_att,
                  ))
        for k in ['all','new','rep']:
            fout.write('{};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f}\n'.format(
                                                        k,
                                                        scores[k]['h01'],
                                                        scores[k]['h05'],
                                                        scores[k]['h10'],
                                                        scores[k]['ndcg01'],
                                                        scores[k]['ndcg05'],
                                                        scores[k]['ndcg10'],
                                                      ))
        if args.model=="BERT": fout.write("mask_prob: %.2f\n"  % (args.mask_prob))
        fout.write('\n')
 

def print_scores(scores):
    for k in ['all','new','rep']:
        print('{}: h@1: {:.5f} h@5: {:.5f} h@10: {:.5f} ndcg@1: {:.5f} ndcg@5: {:.5f} ndcg@10: {:.5f}'.format(
                                                        k,
                                                        scores[k]['h01'],
                                                        scores[k]['h05'],
                                                        scores[k]['h10'],
                                                        scores[k]['ndcg01'],
                                                        scores[k]['ndcg05'],
                                                        scores[k]['ndcg10'],
                                                      ))
    print("ratio: ", scores['ratio'])

def metrics(a):
    a   = np.array(a)
    tot = float(len(a))

    return {
      'h01': (a<1).sum()/tot,
      'h05': (a<5).sum()/tot,
      'h10': (a<10).sum()/tot,
      'ndcg01': np.sum([1 / np.log2(rank + 2) for rank in a[a<1]])/tot,
      'ndcg05': np.sum([1 / np.log2(rank + 2) for rank in a[a<5]])/tot,
      'ndcg10': np.sum([1 / np.log2(rank + 2) for rank in a[a<10]])/tot,
    }

def compute_recall(model, _loader, args, maxit=100000):

    store = {'rrep': [],'rnew': [],'rall': [], 'ratio': []}

    model.eval()
    with torch.no_grad():
        for i,data in tqdm(enumerate(_loader)):
            data = data.to(args.device)
            store = model.compute_rank(data,store,k=10)
            if i>maxit: break

    return {
            'rep': metrics(store['rrep']),
            'new': metrics(store['rnew']),
            'all': metrics(store['rall']),
            'ratio': np.mean(store['ratio']),
           }

def compute_rank(data,store,k=10):
   inputs,pos,_ = convert_batch(data,self.args,sample_neg=False)

   feats = self(inputs,data)
   
   xtsy = torch.zeros_like(pos)
   xtsy[data.x_s_batch,data.x_s[:,3]] = data.xts
   
   if self.args.fr_ctx:
       ctx,batch_inds = self.get_ctx_att(data,inputs,feats)

   if self.args.fr_ctx==False and self.args.fr_rep==True:
       rep_enc = self.rep_emb(self.get_av_rep(inputs,data))
   
   # identify repeated interactions in the batch 
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
 
       if self.args.fr_ctx==False and self.args.fr_rep:         
           # get rep
           reps = inputs[b,inputs[b,:]!=0].unsqueeze(1)==av
           a = (step-xtsy[b,inputs[b,:]!=0]).unsqueeze(1).repeat(1,len(av)) * reps
           if torch.any(torch.any(reps,1)):
               a = a[torch.any(reps,1),:]
               a[a==0]=99999
               a = a.min(0).values*torch.any(reps,0)
               sm  = torch.bucketize(a, self.boundaries)+1
               sm  = sm*torch.any(reps,0)
               sm  = self.rep_emb(sm) 
               av_embs += sm
     
       if self.args.fr_ctx:         
           ctx_expand = torch.zeros(self.args.av_tens.shape[1],self.args.K,device=self.args.device)
           ctx_expand[batch_inds[b,-1,:],:] = ctx[b,-1,:,:]
           scores  = (feats[b,-1,:] * ctx_expand).sum(-1) 
           scores  = scores[:len(av)]
       else:
           scores  = (feats[b,-1,:] * av_embs).sum(-1) 

       iseq = pos[b,-1] == av
       idx  = torch.where(iseq)[0]
       rank = torch.where(torch.argsort(scores, descending=True)==idx)[0].item()

       if mask[b]: # rep
           store['rrep'] += [rank]
       else:
           store['rnew'] += [rank]
       store['rall'] += [rank]
   
   return store


