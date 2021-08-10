from tqdm import tqdm
from sampling import *
from data import *

def train(model,optimizer,_loader,args):
    model.train()

    loss_all = 0.0
    loss_cnt = 0.0
    for data in tqdm(_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
      
        loss = model.train_step(data)
       
        loss_all += loss.item()
        loss_cnt += len(data.y)
        loss.backward()
        optimizer.step()

    return loss_all / loss_cnt


