import sys,os,time
import pandas as pd
from tqdm import tqdm

from arguments import *
from eval import *
from data import *
from models import *

import torch
import torch.optim as optim

args = arg_parse()
print_args(args)
args.device = torch.device(args.device)

MPATH,MODEL = get_model_type(args)

data_fu = load_data(args) 
train_loader, val_loader, test_loader = get_dataloaders(data_fu,args)

# baselines models
if args.model in ['POP','REP']:
    data_tr = data_fu[data_fu.stop<args.pivot_1]
    model = MODEL(args,data_tr)
    scores = compute_recall(model, test_loader, args)
    print("Final score")
    print("="*11)
    print_scores(scores)
# Parametrized models
else:
    model = MODEL(args).to(args.device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    
    best_val = 0.0
    best_max = args.early_stop
    best_cnt = best_max
    
    print("training...")
    for epoch in range(args.num_epochs):
        loss_all = 0.0; loss_cnt = 0
        model.train()
        for data in tqdm(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
    
            loss = model.train_step(data)
            
            loss_all += loss.item()
            loss_cnt += (data[:,:,5]!=0).sum()
            
            loss.backward()
            optimizer.step()
            
            if torch.isnan(loss):
                print("loss is nan !") 
    
        scores = compute_recall(model, val_loader, args, maxit=500)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all/loss_cnt))
        print_scores(scores)
    
        hall = scores['all']['h01']
        if hall>best_val:
            best_val = hall
            torch.save(model.state_dict(), MPATH)
            best_cnt = best_max
        else:
            best_cnt -= 1
            if best_cnt == 0:
                break
    
    model = MODEL(args).to(args.device)
    model.load_state_dict(torch.load(MPATH))
    
    scores = compute_recall(model, test_loader, args)
    print("Final score")
    print("="*11)
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all/loss_cnt))
    print_scores(scores)
    save_scores(scores,args)
 
