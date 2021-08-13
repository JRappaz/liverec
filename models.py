import os
from data import *
from baselines import *

import numpy as np

import torch
import torch.nn as nn

def get_model_type(args):
    # Model and data
    mto  = str(args.mto)
    mto += '_' + str(args.K)
    mto += '_' + str(args.l2)
    mto += '_' + str(args.topk_att)
    mto += '_' + str(args.num_att_ctx)
    mto += '_' + str(args.seq_len)
    mto += "_rep" if args.fr_rep else ""
    mto += "_ctx" if args.fr_ctx else ""
    mto += '.pt'

    if args.model == "POP":
        MODEL = POP
    elif args.model == "REP":
        MODEL = REP
    elif args.model == "MF":
        MODEL = MF
    elif args.model == "FPMC":
        MODEL = FPMC
    elif args.model == "LiveRec":
        MODEL = LiveRec

    return os.path.join(args.model_path,mto),MODEL

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Attention(nn.Module):
    def __init__(self, args, num_att, num_heads, causality=False):
        super(Attention, self).__init__()
        self.args = args
        self.causality = causality

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.K, eps=1e-8)
        
        for _ in range(num_att):
            new_attn_layernorm = nn.LayerNorm(args.K, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(args.K,
                                                    num_heads,
                                                    0.2)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.K, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.K, 0.2)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, timeline_mask=None):
        if self.causality:
            tl = seqs.shape[1] # time dim len for enforce causality
            attention_mask = ~torch.tril(torch.ones((tl, tl), 
                                         dtype=torch.bool, 
                                         device=self.args.device))
        else: attention_mask = None
        
        if timeline_mask != None:
            seqs *= ~timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            if timeline_mask != None:
                seqs *=  ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)


class LiveRec(nn.Module):
    def __init__(self, args):
        super(LiveRec, self).__init__()
        self.args = args

        self.item_embedding = nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.pos_emb = nn.Embedding(args.seq_len, args.K) 
        self.emb_dropout = nn.Dropout(p=0.2)

        # Sequence encoding attention
        self.att = Attention(args, 
                             args.num_att, 
                             args.num_heads, 
                             causality=True)

        # Availability attention
        self.att_ctx = Attention(args, 
                                 args.num_att_ctx, 
                                 args.num_heads_ctx, 
                                 causality=False)

        # Time interval embedding 
        # 24h cycles, except for the first one set to 12h
        self.boundaries = torch.LongTensor([0]+list(range(77,3000+144, 144))).to(args.device)
        self.rep_emb = nn.Embedding(len(self.boundaries)+2, args.K, padding_idx=0)
 
    def forward(self, log_seqs):
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.args.device))

        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0).to(self.args.device)

        feats = self.att(seqs, timeline_mask)

        return feats

    def predict(self,feats,inputs,items,ctx,data):
        if ctx!=None: i_embs = ctx 
        else:         self.item_embedding(items) 

        return (feats * i_embs).sum(dim=-1) 

    def compute_rank(self,data,store,k=10):
        inputs = data[:,:,3] # inputs 
        pos    = data[:,:,5] # targets
        xtsy   = data[:,:,6] # targets ts
 
        feats = self(inputs)
       
        # Add time interval embeddings 
        if self.args.fr_ctx:
            ctx,batch_inds = self.get_ctx_att(data,feats)

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
 
            if self.args.fr_ctx:         
                ctx_expand = torch.zeros(self.args.av_tens.shape[1],self.args.K,device=self.args.device)
                ctx_expand[batch_inds[b,-1,:],:] = ctx[b,-1,:,:]
                scores = (feats[b,-1,:] * ctx_expand).sum(-1) 
                scores = scores[:len(av)]
            else:
                scores = (feats[b,-1,:] * av_embs).sum(-1) 

            iseq = pos[b,-1] == av
            idx  = torch.where(iseq)[0]
            rank = torch.where(torch.argsort(scores, descending=True)==idx)[0].item()

            if mask[b]: # rep
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        
        return store

    def get_ctx_att(self,data,feats,neg=None):
        if not self.args.fr_ctx: return None

        inputs,pos,xtsy = data[:,:,3],data[:,:,5],data[:,:,6] 

        # unbatch indices
        ci = torch.nonzero(inputs, as_tuple=False)
        flat_xtsy = xtsy[ci[:,0],ci[:,1]]

        av = self.args.av_tens[flat_xtsy,:]
        av_embs = self.item_embedding(av)

        # repeat consumption: time interval embeddings
        if self.args.fr_rep:
            av_rep_batch = self.get_av_rep(data)
            av_rep_flat  = av_rep_batch[ci[:,0],ci[:,1]]
            rep_enc = self.rep_emb(av_rep_flat)
            av_embs += rep_enc 

        flat_feats = feats[ci[:,0],ci[:,1],:]
        flat_feats = flat_feats.unsqueeze(1).expand(flat_feats.shape[0],
                                                    self.args.av_tens.shape[-1],
                                                    flat_feats.shape[1])

        scores = (av_embs * flat_feats).sum(-1)
        inds   = scores.topk(self.args.topk_att,dim=1).indices
        
        # embed selected items
        seqs = torch.gather(av_embs, 1, inds.unsqueeze(2) \
                    .expand(-1,-1,self.args.K))

        seqs = self.att_ctx(seqs)

        def expand_att(items):
            av_pos = torch.where(av==items[ci[:,0],ci[:,1]].unsqueeze(1))[1]
            is_in = torch.any(inds == av_pos.unsqueeze(1),1)
            
            att_feats = torch.zeros(av.shape[0],self.args.K).to(self.args.device)
            att_feats[is_in,:] = seqs[is_in,torch.where(av_pos.unsqueeze(1) == inds)[1],:]
            
            out = torch.zeros(inputs.shape[0],inputs.shape[1],self.args.K).to(self.args.device)
            out[ci[:,0],ci[:,1],:] = att_feats
            return out

        # training
        if pos != None and neg != None:
            return expand_att(pos),expand_att(neg)
        # testing
        else:
            out = torch.zeros(inputs.shape[0],inputs.shape[1],seqs.shape[1],self.args.K).to(self.args.device)
            out[ci[:,0],ci[:,1],:] = seqs
            batch_inds = torch.zeros(inputs.shape[0],inputs.shape[1],inds.shape[1],dtype=torch.long).to(self.args.device)
            batch_inds[ci[:,0],ci[:,1],:] = inds
            return out,batch_inds

    def train_step(self, data, use_ctx=False): # for training
        inputs,pos = data[:,:,3],data[:,:,5]
        neg = sample_negs(data,self.args).to(self.args.device)

        feats = self(inputs)
      
        ctx_pos,ctx_neg = None,None
        if self.args.fr_ctx:
            ctx_pos,ctx_neg = self.get_ctx_att(data,feats,neg)
            
        pos_logits = self.predict(feats,inputs,pos,ctx_pos,data) 
        neg_logits = self.predict(feats,inputs,neg,ctx_neg,data) 

        loss  = (-torch.log(pos_logits[inputs!=0].sigmoid()+1e-24)
                 -torch.log(1-neg_logits[inputs!=0].sigmoid()+1e-24)).sum()

        return loss

    def get_av_rep(self,data):
        bs     = data.shape[0]
        inputs = data[:,:,3] # inputs 
        xtsb   = data[:,:,2] # inputs ts
        xtsy   = data[:,:,6] # targets ts

        av_batch  = self.args.av_tens[xtsy.view(-1),:]
        av_batch  = av_batch.view(xtsy.shape[0],xtsy.shape[1],-1)
        av_batch *= (xtsy!=0).unsqueeze(2) # masking pad inputs
        av_batch  = av_batch.to(self.args.device)

        mask_caus = 1-torch.tril(torch.ones(self.args.seq_len,self.args.seq_len),diagonal=-1)
        mask_caus = mask_caus.unsqueeze(0).unsqueeze(3)
        mask_caus = mask_caus.expand(bs,-1,-1,self.args.av_tens.shape[-1])
        mask_caus = mask_caus.type(torch.bool).to(self.args.device)
       
        tile = torch.arange(self.args.seq_len).unsqueeze(0).repeat(bs,1).to(self.args.device)
 
        bm   = (inputs.unsqueeze(2).unsqueeze(3)==av_batch.unsqueeze(1).expand(-1,self.args.seq_len,-1,-1))
        bm  &= mask_caus

        # **WARNING** this is a hacky way to get the last non-zero element in the sequence.
        # It works with pytorch 1.8.1 but might break in the future. 
        sm   = bm.type(torch.int).argmax(1)
        sm   = torch.any(bm,1) * sm
        
        sm   = (torch.gather(xtsy, 1, tile).unsqueeze(2) - 
                torch.gather(xtsb.unsqueeze(2).expand(-1,-1,self.args.av_tens.shape[-1]), 1, sm))
        sm   = torch.bucketize(sm, self.boundaries)+1
        sm   = torch.any(bm,1) * sm
        
        sm  *= av_batch!=0
        sm  *= inputs.unsqueeze(2)!=0
        return sm


