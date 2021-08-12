import argparse
from prettytable import PrettyTable

def print_args(parse_args):
    x = PrettyTable()
    x.field_names = ["Arg.", "Value"]
    for arg in vars(parse_args):
        x.add_row([arg, getattr(parse_args, arg)])
    print(x)

def arg_parse():
        parser = argparse.ArgumentParser(description='LiveRec - Twitch')

        parser.add_argument('--seed', dest='seed', type=int,
            help='Random seed')

        parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size - only active if torch is used')
        parser.add_argument('--seq_len', dest='seq_len', type=int,
            help='Max size of the sequence to consider')

        parser.add_argument('--num_heads', dest='num_heads', type=int,
            help='Numer of heads to use for multi-heads attention')
        parser.add_argument('--num_heads_ctx', dest='num_heads_ctx', type=int,
            help='Numer of heads to use for multi-heads attention CTX')


        parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
 
        parser.add_argument('--model', dest='model', type=str,
            help='Type of the model')

        parser.add_argument('--model_from', dest='mfrom', type=str,
            help='Name of the model to load')
        parser.add_argument('--model_to', dest='mto', type=str,
            help='Name of the model to save')
        parser.add_argument('--cache_dir', dest='cache_dir', type=str,
            help='Path to save the cached preprocessd dataset')
 
        parser.add_argument('--model_path', dest='model_path', type=str,
            help='Path to save the model')
        parser.add_argument('--early_stop', dest='early_stop', type=int,
            help='Number of iteration without improvment before stop')
        parser.add_argument('--ev_sample', dest='ev_sample', type=int,
            help='Number of samples for the final evaluation')
        parser.add_argument('--device', dest='device', type=str,
            help='Pytorch device')

        parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
        parser.add_argument('--mask_prob', dest='mask_prob', type=float,
            help='BERT mask prob.')
        parser.add_argument('--l2', dest='l2', type=float,
            help='Strength of L2 regularization')
        parser.add_argument('--dim', dest='K', type=int,
            help='Number of latent factors')
 
        parser.add_argument('--num_iters', dest='num_iter', type=int,
            help='Number of training iterations')
        parser.add_argument('--num_epochs', dest='num_epochs', type=int,
            help='Number of training epochs')
        parser.add_argument('--num_att', dest='num_att', type=int,
            help='Num attention module for seq encoding')
        parser.add_argument('--num_att_ctx', dest='num_att_ctx', type=int,
            help='Num attention for ctx module')
 
        parser.add_argument('--topk_att', dest='topk_att', type=int,
            help='Items to send to attentive output')

        parser.add_argument('--fr_ctx', dest='fr_ctx', nargs='?',
            const=True, default=False,
            help='')
        parser.add_argument('--fr_rep', dest='fr_rep', nargs='?',
            const=True, default=False,
            help='')
        parser.add_argument('--uniform', dest='uniform', nargs='?',
            const=True, default=False,
            help='')
        parser.add_argument('--debug', dest='debug', nargs='?',
            const=True, default=False,
            help='')
        parser.add_argument('--caching', dest='caching', nargs='?',
            const=True, default=False,
            help='')

        parser.set_defaults(
                        seed=42,
                        dataset="/mnt/localdata/rappaz/twitch/data/v3/100k/",
                        lr=0.0005,
                        l2=0.1,  
                        mask_prob=0.5,  
                        batch_size=100, 
                        num_att=2,
                        num_att_ctx=2,
                        num_heads=4,
                        num_heads_ctx=4,
                        num_iter=200,
                        seq_len=16,  
                        topk_att=64,  
                        early_stop=15,  
                        K=64,  
                        num_epochs=150,
                        model="LiveRec",
                        model_path="/mnt/datastore/rappaz/twitch/models",
                        mto="liverec",
                        device="cuda",
                        cache_dir="dataset/"
                       )

        args = parser.parse_args()
        return args 
