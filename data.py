import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sampling import *

import torch.utils.data as data
from torch.utils.data import DataLoader

def load_data(args):
    INFILE = os.path.join(args.dataset,'100k.csv')
    #user,stream,streamer_id,start,stop
    cols = ["user","stream","streamer","start","stop"]
    data_fu = pd.read_csv(INFILE, header=None, names=cols)
    
    # Add one for padding
    data_fu.user = pd.factorize(data_fu.user)[0]+1
    data_fu['streamer_raw'] = data_fu.streamer
    data_fu.streamer = pd.factorize(data_fu.streamer)[0]+1
    print("Num users: ", data_fu.user.nunique())
    print("Num streamers: ", data_fu.streamer.nunique())
    print("Num interactions: ", len(data_fu))
    print("Estimated watch time: ", (data_fu['stop']-data_fu['start']).sum() * 5 / 60.0)
    
    args.M = data_fu.user.max()+1 # users
    args.N = data_fu.streamer.max()+2 # items
    
    data_temp = data_fu.drop_duplicates(subset=['streamer','streamer_raw'])
    umap      = dict(zip(data_temp.streamer_raw.tolist(),data_temp.streamer.tolist()))
    
    # Splitting and caching
    max_step = max(data_fu.start.max(),data_fu.stop.max())
    print("Num timesteps: ", max_step)
    args.max_step = max_step
    args.pivot_1  = max_step-500
    args.pivot_2  = max_step-250
    
    print("caching availability")
    ts = {}
    max_avail = 0
    for s in range(max_step+1):
        all_av = data_fu[(data_fu.start<=s) & (data_fu.stop>s)].streamer.unique().tolist()
        ts[s] = all_av
        max_avail = max(max_avail,len(ts[s]))
    args.max_avail = max_avail
    args.ts = ts
    print("max_avail: ", max_avail)
    
    # Compute availability matrix of size (num_timesteps x max_available)
    max_av   = max([len(v) for k,v in args.ts.items()])
    max_step = max([k for k,v in args.ts.items()])+1
    av_tens = torch.zeros(max_step,max_av).type(torch.long)
    for k,v in args.ts.items():
        av_tens[k,:len(v)] = torch.LongTensor(v)
    args.av_tens = av_tens.to(args.device)
    return data_fu

def get_dataloaders(data_fu, args):
    if args.debug:
        mu = 1000
    else:
        mu = int(10e9)
 
    cache_tr = os.path.join(args.cache_dir,"100k_tr.p")
    cache_te = os.path.join(args.cache_dir,"100k_te.p")
    cache_va = os.path.join(args.cache_dir,"100k_val.p")

    if args.caching and all(list(map(os.path.isfile,[cache_tr,cache_te,cache_va]))):
        datalist_tr = pickle.load(open(cache_tr, "rb"))
        datalist_va = pickle.load(open(cache_va, "rb"))
        datalist_te = pickle.load(open(cache_te, "rb"))
    elif args.caching: 
        datalist_tr = get_sequences(data_fu,0,args.pivot_1,args,mu)
        datalist_va = get_sequences(data_fu,args.pivot_1,args.pivot_2,args,mu)
        datalist_te = get_sequences(data_fu,args.pivot_2,args.max_step,args,mu)

        pickle.dump(datalist_te, open(cache_te, "wb"))
        pickle.dump(datalist_tr, open(cache_tr, "wb"))
        pickle.dump(datalist_va, open(cache_va, "wb"))

    train_loader = DataLoader(datalist_tr,batch_size=args.batch_size,
                              collate_fn=lambda x: custom_collate(x,args))
    val_loader   = DataLoader(datalist_va,batch_size=args.batch_size,
                              collate_fn=lambda x: custom_collate(x,args))
    test_loader  = DataLoader(datalist_te,batch_size=args.batch_size,
                              collate_fn=lambda x: custom_collate(x,args))

    return train_loader, val_loader, test_loader


def custom_collate(batch,args):
    # returns a [batch x seq x feats] tensor
    # feats: [padded_positions,positions,inputs_ts,items,users,targets,targets_ts]

    bs = len(batch)
    feat_len = len(batch[0])
    batch_seq = torch.zeros(bs,args.seq_len, feat_len, dtype=torch.long)
    for ib,b in enumerate(batch):
        for ifeat,feat in enumerate(b):
            batch_seq[ib,b[0],ifeat] = feat
    return batch_seq

class SequenceDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask

def get_sequences(_data, _p1, _p2, args, max_u=int(10e9)):
    data_list = []

    _data = _data[_data.stop<_p2].copy()
    
    grouped = _data.groupby('user')
    for user_id, group in tqdm(grouped):
        group = group.sort_values('start')
        group = group.tail(args.seq_len+1)
        if len(group)<2: continue

        group = group.reset_index(drop=True) 
        
        # Get last interaction
        last_el = group.tail(1)
        yt = last_el.start.values[0]
        group.drop(last_el.index,inplace=True)

        # avoid including train in test/validation
        if yt < _p1 or yt >= _p2: continue

        padlen = args.seq_len - len(group)

        # sequence input features
        positions  = torch.LongTensor(group.index.values)
        inputs_ts  = torch.LongTensor(group.start.values)
        items      = torch.LongTensor(group['streamer'].values)
        users      = torch.LongTensor(group.user.values)
        bpad       = torch.LongTensor(group.index.values + padlen)

        # sequence output features
        targets    = torch.LongTensor(items[1:].tolist() + [last_el.streamer.values[0]])
        targets_ts = torch.LongTensor(inputs_ts[1:].tolist() + [last_el.start.values[0]])

        data_list.append([bpad,positions,inputs_ts,items,users,targets,targets_ts])

        # stop if user limit is reached
        if len(data_list)>max_u: break

    return SequenceDataset(data_list)


