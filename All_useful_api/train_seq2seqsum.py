import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import GigaDataset, prepro_batch
from source.Seq2Seq_beifen import Seq2SeqSum
#from source.CopySeq2Seq import CopySeq2SeqSum
from source.training import Trainer
from source.utils import make_word2id

#For Training

def main(emb_size,hidden_size):
    #get args
    parser = argparse.ArgumentParser(description="Seq2SeqSum Training Program")
    #model args
    parser.add_argument("--vocab_size", type=int, default=94) #去空格，还有94个可打印字符
    parser.add_argument("--emb_dim", type=int, default=emb_size) #128
    parser.add_argument("--n_hidden", type=int, default=hidden_size) #256
    parser.add_argument("--n_layer", type=int, default=3)
    # parser.add_argument("--kernel_size", type=int, default=[1,3,5])
    # parser.add_argument("--filter_size", type=int, default=128)
    parser.add_argument('--copy', dest='copy', action='store_true')
    parser.add_argument('--no-copy', dest='copy', action='store_false')
    parser.set_defaults(copy=False)

    parser.add_argument("--data_path", type=str,
                        default="tmp")
    parser.add_argument("--max_src_len", type=int, default=16)
    parser.add_argument("--max_tgt_len", type=int, default=16)

    #training argsparser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--vocab_path", type=str, default="tmp/vocab.pkl")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--ckpt_freq", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)

    args = parser.parse_args()

    # make vocab
    word2id = make_word2id(args.vocab_path, args.vocab_size)
    #print(word2id)
    # id2word = {value:key for key, value in word2id.items()}
    vsize = len(word2id)
    # init data loader
    train_loader = DataLoader(
        GigaDataset(args.data_path, 'train'),
        batch_size=args.batch_size,
        collate_fn=partial(
            prepro_batch, args.max_src_len,
            args.max_tgt_len, word2id
            )
        )


    val_loader = DataLoader(
        GigaDataset(args.data_path, 'val'),
        batch_size=args.batch_size,
        collate_fn=partial(
            prepro_batch, args.max_src_len,
            args.max_tgt_len, word2id
            )
        )

    

    # init model
    use_cuda = 'YES' if torch.cuda.is_available() else 'NOPE'
    #use_cuda = False # 放到服务器上用
    #use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Are u using CUDA for ACCELERATE?',use_cuda)
    # if not args.copy:
    #print("Warning: training a model without copy mechanism!")
    model = Seq2SeqSum(
            vsize, args.emb_dim,
            args.n_hidden, args.n_layer
            ).to(device)
    # else:
    #     model = CopySeq2SeqSum(
    #         vsize, args.emb_dim,
    #         args.n_hidden, args.n_layer
    #         ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(optimizer, model, train_loader,
                      val_loader, args.save_dir, device,
                      args.clip, args.print_freq, args.ckpt_freq,
                      args.patience, args.copy)
    trainer.train()

if __name__ == "__main__":
    pass
