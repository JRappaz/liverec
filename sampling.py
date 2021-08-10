import random
import torch

def sample_av(p,t,args):
    # availability sampling
    av = args.ts[t]
    while True:
        ridx = random.randint(0,len(av)-1)
        ri   = av[ridx]
        if p!=ri:
            return ri
 
def sample_uni(p,t,args):
    # uniform sampling
    while True:
        ri = random.randint(0,args.N-1)
        if p!=ri:
            return ri

def sample_negs(data,args):
    pos,xts = data[:,:,5],data[:,:,6]
    neg = torch.zeros_like(pos)

    ci = torch.nonzero(pos, as_tuple=False)

    for i in range(ci.shape[0]):
        p = pos[ci[i,0],ci[i,1]].item()
        t = xts[ci[i,0],ci[i,1]].item()

        if args.uniform:
            neg[ci[i,0],ci[i,1]] = sample_uni(p,t,args)
        else:
            neg[ci[i,0],ci[i,1]] = sample_av(p,t,args)

    return neg

