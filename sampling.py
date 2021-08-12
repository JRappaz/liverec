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
    ps = pos[ci[:,0],ci[:,1]].tolist()
    ts = xts[ci[:,0],ci[:,1]].tolist()

    for i in range(ci.shape[0]):
        p = ps[i]; t = ts[i]

        if args.uniform:
            neg[ci[i,0],ci[i,1]] = sample_uni(p,t,args)
        else:
            neg[ci[i,0],ci[i,1]] = sample_av(p,t,args)

    return neg

