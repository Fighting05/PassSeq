import string

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random
import math

#4.7303640959971275e-06   4.73036425319151e-06
class Seq2SeqSum(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, n_layer=1, bi_enc=True, dropout=0.0):
    # def __init__(self, vocab_size, emb_dim,kernel_size,filter_size,
    #              n_hidden, n_layer=1, bi_enc=True, dropout=0.2):
        super(Seq2SeqSum, self).__init__()

        self.n_layer = n_layer
        self.bi_enc = bi_enc  # whether encoder is bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # self.kernel_size = kernel_size
        # self.filter_size = filter_size
        # self.conv1 = nn.Conv1d(emb_dim,self.filter_size,self.kernel_size[0],stride=1,padding=0)
        # self.conv2 = nn.Conv1d(emb_dim,self.filter_size,self.kernel_size[1],stride=1,padding=1)
        # self.conv3 = nn.Conv1d(emb_dim,self.filter_size,self.kernel_size[2],stride=1,padding=2)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional = bi_enc,
            dropout = 0 if n_layer==1 else dropout
        )

        #initial encoder hidden states as learnable parameters
        states_size0 = n_layer * (2 if bi_enc else 1)
        self.enc_init_h = nn.Parameter(
            torch.Tensor(states_size0, n_hidden)
        )
        self.enc_init_c = nn.Parameter(
            torch.Tensor(states_size0, n_hidden)
        )
        init.uniform_(self.enc_init_h, -1e-2, 1e-2)
        init.uniform_(self.enc_init_c, -1e-2, 1e-2)

        #reduce encoder states to decoder initial states
        self.enc_out_dim = n_hidden * (2 if bi_enc else 1)
        self._dec_h = nn.Linear(self.enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(self.enc_out_dim, n_hidden, bias=False)

        self.decoder = AttnDecoder(
            self.embedding, n_hidden, vocab_size,
            self.enc_out_dim, n_layer,
            dropout=dropout
            )

        #######################################################
        self.weak_pwds = []
        self.weak_pros = []

        self.dfsrng = {0:31,1:31,2:31,3:31,4:31}

        self.dfsscore = {0: 1,
                     1: 2, 2: 2, 3: 2,
                     4: 3, 5: 3,
                     6: 4, 7: 4, 8: 4, 9: 4, 10: 4,
                     11: 5, 12: 5, 13: 5, 14: 5, 15: 5,
                     16: 6, 17: 6, 18: 6, 19: 6, 20: 6,
                     21: 7, 22: 7, 23: 7, 24: 7, 25: 7,
                     26: 8, 27: 8, 28: 8, 29: 8, 30: 8,
                     31: 9, 32: 9, 33: 9, 34: 9, 35: 9,
                     36: 10, 37: 10, 38: 10, 39: 10, 40: 10}
        #######################################################

    def forward(self, src, src_lengths, tgt):
        """args:
            src: [batch_size, max_len]
            src_lengths: [batch_size]
            tgt: [batch_size, max_len]
        """
        enc_outs, init_dec_states = self.encode(src, src_lengths)
        attn_mask = len_mask(src_lengths).to(src.device)
        assert attn_mask.device == src.device
        logit = self.decoder(tgt, init_dec_states, enc_outs, attn_mask)
        #return logit: [batch_size, max_len, voc_size]
        return logit

    def encode(self, src, src_lengths):
        """run encoding"""
        #print(src_lengths)
        #expand init encoder states in batch size dim
        size = (
            self.enc_init_c.size(0),
            len(src_lengths),
            self.enc_init_c.size(1)
        )
        init_hidden = (
            self.enc_init_h.unsqueeze(1).expand(*size).contiguous(),
            self.enc_init_c.unsqueeze(1).expand(*size).contiguous()
        )

        #print('encode src',src)
        embed = self.embedding(src.transpose(0, 1))
        #print('encode embed',embed)
        padded_seq = pack_padded_sequence(embed, src_lengths.cpu())
        #print('encode padded_seq', padded_seq)
        #print('encode init_hidden', init_hidden)

        enc_out, hidden = self.encoder(padded_seq, init_hidden)
        #print('encode enc_outs',enc_out)
        outputs, _ = pad_packed_sequence(enc_out)
        init_dec_states_list = []
        #init dec_input and hidden
        if self.bi_enc:
            h, c = hidden

            for i in range(self.n_layer):
                h_into_cell = h.chunk(self.n_layer, dim=0)[i].view(1,h.size(1),-1)
                c_into_cell = c.chunk(self.n_layer, dim=0)[i].view(1,c.size(1),-1)
                init_dec_states_list.append((self._dec_h(h_into_cell).squeeze(0),self._dec_c(c_into_cell).squeeze(0)))

        #print('init_dec_states:',init_dec_states)
        return outputs, init_dec_states_list

#############################################################################################该分割线以上请不要随意修改
    def make_new_smf(self,sfm,command=''):
        len_sfm = len(sfm)
        normal_sfm = [0.0 for _ in range(len_sfm)]

        total_f = sum(sfm[5:]) + sfm[3]  # 91 4是*; 3是<end>
        total_f = total_f.item()

        for k in range(len_sfm):
            if k in [0, 1, 2, 4]:
                normal_sfm[k] = 0.0
            else:
                normal_sfm[k] = sfm[k].item() / total_f
        if command=='normal':
            return normal_sfm
        # print('normal_sfm',normal_sfm)
        cdf_sfm = [0.0 for _ in range(len_sfm + 1)]
        lf = 0.0
        len_normal_sfm = len(normal_sfm)
        for i in range(0, len_normal_sfm):  # 前四个是XXX
            cdf_sfm[i] = lf
            lf += normal_sfm[i]
        cdf_sfm[len_normal_sfm] = lf

        return cdf_sfm,normal_sfm


    def pre_calculate(self,inps,word2id):#inps就是模板  #总纲算法，预处理用的
        id2word = dict((id_, word) for word, id_ in word2id.items())
        #print('pre_calculate:inp', inps)
        #pre_calculate:inp * u * s h * n *
        inp = torch.LongTensor([[word2id.get(word, word2id['*'])
                                 for word in inps.split()[:]]])
        #print('inps,',inp)
        #inps, tensor([[ 4, 22,  4, 12, 21,  4, 11,  4]])
        inp_len = torch.LongTensor([inp.size(1)])
        #print('inp_len, ',inp_len)
        #inp_len, tensor([8])
        attn_mask = torch.ones_like(inp).long()
        #print('pre_calculate:attn_mask', attn_mask)
        #pre_calculate:attn_mask tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
        SOS, END = word2id['<start>'], word2id['<end>']
        #print('SOS, END',SOS, END)
        #SOS, END 2 3
        # store top k sequence score, init it as zero
        pwd_pro = 1.0 #乘法
        #print('pre_calculate:top_k_scores', top_k_scores)
        #pre_calculate:top_k_scores tensor([0.])
        # store top k squence
        top_k_words = torch.ones([1]).long() * SOS
        #print('pre_calculate:top_k_words', top_k_words)
        #pre_calculate:top_k_words tensor([2])
        # store completed seqs and their scores
        prev_words = top_k_words
        vocab_size = len(word2id)
        #print('pre_calculate:vocab_size', vocab_size)
        #pre_calculate:vocab_size 94

        # encoding
        enc_outs, hc_list = self.encode(inp, inp_len)
        #print('pre hc_list', hc_list)
        #print('pre enc_outs', enc_outs)
        hc_list_new = []
        for i in range(len(hc_list)):
            h = hc_list[i][0].expand(1, hc_list[i][0].size(1))
            c = hc_list[i][1].expand(1, hc_list[i][1].size(1))
            hc_list_new.append((h, c))
        #print('hc_list_new', hc_list_new)
        return prev_words,hc_list_new,enc_outs,attn_mask,pwd_pro,vocab_size,id2word #传到下面这个函数中

    #
    # def mask_attack_search(self, prev_words,hc_list_new,enc_outs,attn_mask,pwd_pro,vocab_size,id2word
    #                        ,pwd_tp,i,mask_i,threshold_min,word2id):#inp是加入了空格的模板,pwd_tp是普通的模板加'\n'
    #
    #     mask_i2k = {1:40,2:40,3:40,4:40,5:40}
    #
    #
    #     if len(self.weak_pwds)==100001:
    #         return #这一句话先注释掉
    #     it = pwd_tp[i]
    #
    #     if it!='*':
    #         if it!='\n':
    #             dec_out, hc_list_new = self.decoder._step(
    #                 prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
    #             logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
    #             sfm = logit.squeeze(0)
    #             normal_sfm = self.make_new_smf(sfm,'normal')
    #             index_ = word2id[it]
    #             pwd_pro_new = pwd_pro*normal_sfm[index_]
    #             index_ = torch.tensor(index_)
    #
    #             if pwd_pro_new>threshold_min:
    #                 prev_words_new = torch.cat([prev_words, index_.unsqueeze(0)], dim=0)
    #                 self.mask_attack_search(prev_words_new,hc_list_new,enc_outs,attn_mask,pwd_pro_new,vocab_size,id2word
    #                        ,pwd_tp,i+1,mask_i,threshold_min,word2id)
    #             else:
    #                 return
    #         else:#it=='\n'
    #             dec_out, hc_list_new = self.decoder._step(
    #                     prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
    #             logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
    #             sfm = logit.squeeze(0)
    #             normal_sfm = self.make_new_smf(sfm, 'normal')
    #             index_ = word2id['<end>']
    #             #index_ = torch.tensor(index_)
    #             pwd_pro_new = pwd_pro * normal_sfm[index_]
    #             if pwd_pro_new>=threshold_min:
    #                 #prev_words = torch.cat([prev_words, index_.unsqueeze(0)], dim=0)
    #                 pwd = ''
    #                 for k in prev_words:
    #                     pwd += id2word[k.item()]
    #
    #                 print(pwd[7:],str(pwd_pro_new.item()))#######################################
    #                 self.weak_pwds.append(pwd[7:])
    #                 self.weak_pros.append(pwd_pro_new)
    #
    #     else:#it=='*
    #         top_k = mask_i2k[mask_i]
    #
    #         dec_out, hc_list_new = self.decoder._step(
    #                 prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
    #         logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
    #         sfm = logit.squeeze(0)
    #         normal_sfm = self.make_new_smf(sfm,'normal')
    #         #print('normal_sfm', normal_sfm)
    #
    #         # current time step topk
    #         normal_sfm = torch.tensor(normal_sfm)
    #         #print('normal_sfm',normal_sfm)
    #         top_k_pro,top_k_words = normal_sfm.topk(k=top_k, dim=0)
    #         #print('top_k_words',top_k_words)
    #         # print('top_k_scores',top_k_scores,'ctop_k_words',ctop_k_words)
    #
    #         for q in range(len(top_k_words)):
    #
    #             next_word_inds = top_k_words[q]  #这是个tensor！！！ ，所以下面的next_word_inds.unsqueeze(0)能用，要把普通数字转成tensor
    #             if next_word_inds!=3:
    #                 pwd_pro_new = pwd_pro*normal_sfm[next_word_inds]
    #                 if pwd_pro_new < threshold_min:
    #                     continue
    #                 else:
    #                     prev_words_new = torch.cat([prev_words, next_word_inds.unsqueeze(0)], dim=0)
    #                     self.mask_attack_search(prev_words_new,hc_list_new,enc_outs,attn_mask,pwd_pro_new,vocab_size,id2word
    #                            ,pwd_tp,i+1,mask_i+1,threshold_min,word2id)

    """
    self.dfsrng = {0:51,1:51,2:51,3:51,4:51}

    self.dfsscore = {0: 1,
                     1: 2, 2: 2, 3: 2,
                     4: 3, 5: 3,
                     6: 4, 7: 4, 8: 4, 9: 4, 10: 4,
                     11: 5, 12: 5, 13: 5, 14: 5, 15: 5,
                     16: 6, 17: 6, 18: 6, 19: 6, 20: 6,
                     21: 7, 22: 7, 23: 7, 24: 7, 25: 7,
                     26: 8, 27: 8, 28: 8, 29: 8, 30: 8,
                     31: 9, 32: 9, 33: 9, 34: 9, 35: 9,
                     36: 10, 37: 10, 38: 10, 39: 10, 40: 10}
    """



    def mask_attack_Asearch(self, prev_words,hc_list_new,enc_outs,attn_mask,pwd_pro,vocab_size,id2word
                           ,pwd_tp,i,threshold,word2id,FF):#inp是加入了空格的模板,pwd_tp是普通的模板加'\n'


        # if len(self.weak_pwds)==100001:
        #     return

        it = pwd_tp[i]

        if it!='*':
            if it!='\n':
                dec_out, hc_list_new = self.decoder._step(
                    prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
                logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
                sfm = logit.squeeze(0)
                normal_sfm = self.make_new_smf(sfm,'normal')
                index_ = word2id[it]
                pwd_pro_new = pwd_pro*normal_sfm[index_]
                index_ = torch.tensor(index_)

                if pwd_pro_new>threshold:
                    prev_words_new = torch.cat([prev_words, index_.unsqueeze(0)], dim=0)
                    self.mask_attack_Asearch(prev_words_new,hc_list_new,enc_outs,attn_mask,pwd_pro_new,vocab_size,id2word
                           ,pwd_tp,i+1,threshold,word2id,FF)
                else:
                    return
            else:#it=='\n'
                dec_out, hc_list_new = self.decoder._step(
                        prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
                logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
                sfm = logit.squeeze(0)
                normal_sfm = self.make_new_smf(sfm, 'normal')
                index_ = word2id['<end>']
                #index_ = torch.tensor(index_)
                pwd_pro_new = pwd_pro * normal_sfm[index_]
                if pwd_pro_new>threshold:
                    #prev_words = torch.cat([prev_words, index_.unsqueeze(0)], dim=0)
                    pwd = ''
                    for _k in prev_words:
                        pwd += id2word[_k.item()]

                    print(pwd[7:],str(pwd_pro_new.item()))#######################################
                    # self.weak_pwds.append(pwd[7:])
                    # self.weak_pros.append(pwd_pro_new)
                    FF.write(pwd[7:]+'\n')

        else:#it=='*
            dec_out, hc_list_new = self.decoder._step(
                    prev_words[-1].unsqueeze(0), hc_list_new, enc_outs, attn_mask)
            logit = F.softmax(dec_out, dim=1)  # [k, vocab_size]
            sfm = logit.squeeze(0)
            normal_sfm = self.make_new_smf(sfm,'normal')
            #print('normal_sfm', normal_sfm)

            # current time step topk
            normal_sfm = torch.tensor(normal_sfm)
            #print('normal_sfm',normal_sfm)
            t_k = 20
            top_k_pro,top_k_words = normal_sfm.topk(k=t_k, dim=0)
            #print('top_k_words',top_k_words)
            # print('top_k_scores',top_k_scores,'ctop_k_words',ctop_k_words)

            for q in range(t_k):

                next_word_inds = top_k_words[q]  #这是个tensor！！！ ，所以下面的next_word_inds.unsqueeze(0)能用，要把普通数字转成tensor
                if next_word_inds==3:
                    continue
                pwd_pro_new = pwd_pro*normal_sfm[next_word_inds]
                if pwd_pro_new < threshold:
                    continue
                else:
                    prev_words_new = torch.cat([prev_words, next_word_inds.unsqueeze(0)], dim=0)
                    self.mask_attack_Asearch(prev_words_new,hc_list_new,enc_outs,attn_mask,pwd_pro_new,vocab_size,id2word
                               ,pwd_tp,i+1,threshold,word2id,FF)



    def mask_attack_A(self,inps,word2id,pwd_tp, i, mask_i,score,score2,threshold,FF):
        prev_words, hc_list_new, enc_outs, attn_mask, \
        pwd_pro, vocab_size, id2word = self.pre_calculate(inps, word2id)

        self.mask_attack_Asearch( prev_words, hc_list_new, enc_outs, attn_mask, pwd_pro, vocab_size, id2word
                                , pwd_tp, i,threshold,word2id,FF)








    # def mask_attack(self,inps,word2id,pwd_tp, i, threshold_min):
    #     prev_words, hc_list_new, enc_outs, attn_mask, \
    #     pwd_pro, vocab_size, id2word = self.pre_calculate(inps, word2id)
    #
    #     self.mask_attack_search( prev_words, hc_list_new, enc_outs, attn_mask, pwd_pro, vocab_size, id2word
    #                             , pwd_tp, i, 1, threshold_min,word2id)









    def pwd_probability_normal(self, pwd, pwd_tp, word2id): #把不用的概率去掉然后平滑之后的 pr[pwd|conditional]
        #print(word2id)
        #print(pwd)
        pwd_pro = 1.0
        pwd_tensor = torch.LongTensor([[word2id.get(word, word2id['<unk>'])
                                            for word in pwd]])
        #pwd_tensor = torch.LongTensor([[word2id[word]]for word in pwd])
        #print('pwd_tensor',pwd_tensor)
        pwd_tensor = pwd_tensor.squeeze(0)
        #print('pwd_tensor', pwd_tensor)
        SOS, END = word2id['<start>'], word2id['<end>']
        SOS_tensor = torch.ones([1]).long() * SOS
        END_tensor = torch.ones([1]).long() * END
        pwd_tensor = torch.cat([SOS_tensor, pwd_tensor], dim=0)
        pwd_tensor = torch.cat([pwd_tensor, END_tensor], dim=0)
        #print('pwd_tensor:', pwd_tensor)
        prev_word, hc_list_new, enc_outs, attn_mask, \
        top_k_scores, vocab_size, id2word = self.pre_calculate(pwd_tp, word2id)
        # print('外部hc_list_new', hc_list_new)

        for i in range(len(pwd_tensor)-1):
            dec_out, hc_list_new = self.decoder._step(
                pwd_tensor[i].unsqueeze(0), hc_list_new, enc_outs, attn_mask)

            sfm = F.softmax(dec_out, dim=1)  # [k, vocab_size]
            sfm = sfm.squeeze(0)
            normal_sfm = self.make_new_smf(sfm,'normal')

            #print(len(normal_sfm),normal_sfm)
            #print('normal_sfm[pwd_tensor[i + 1]]',normal_sfm[pwd_tensor[i + 1]],pwd_tensor[i + 1])
            pwd_pro*=normal_sfm[pwd_tensor[i + 1]]
            #print('it-->', pwd_tensor[i])
                # print('logit',logit)
                # print('pwd_pro',pwd_pro)
                # print('logit', logit)

        return pwd_pro







    # def comb(self,inp,word2id,thres):
    #     prev_word, hc_list_new, enc_outs, attn_mask,\
    #     top_k_scores,vocab_size,id2word = self.pre_calculate(inp,word2id)
    #     self.dfs_search(prev_word, hc_list_new, enc_outs,
    #                     attn_mask,top_k_scores,vocab_size,id2word,'out2.txt',thres)



#############################################################################################该分割线以下请不要随意修改

class AttnDecoder(nn.Module):
    def __init__(self, embedding, hidden_size,
                 output_size, enc_out_dim, n_layers=1, dropout=0.1):
        super(AttnDecoder, self).__init__()

        self.embedding = embedding
        self.n_layers = n_layers
        self.rest_decoder_cells = []
        emb_size = embedding.weight.size(1)
        self.decoder_cell = nn.LSTMCell(emb_size, hidden_size)
        self.decoder_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder_cell3 = nn.LSTMCell(hidden_size, hidden_size)
        #self.decoder_cell# = nn.LSTMCell(hidden_size, hidden_size)
        self.attn = nn.Linear(enc_out_dim, hidden_size)
        self.concat = nn.Linear(enc_out_dim+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, target, init_states, enc_outs, attn_mask):
        max_len = target.size(1)
        states = init_states
        logits = []
        #print('一次性的states',states)
        for i in range(max_len):
            #the i step target: [batch_size, 1]
            target_i = target[:, i:i+1]
            #one step decoding, use teacher forcing
            #import pdb;pdb.set_trace()
            logit, states = self._step(target_i, states, enc_outs, attn_mask)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)

        return logits

    def _step(self, inp, last_hidden_list, enc_outs, attn_mask):
        embed = self.embedding(inp).squeeze(1)
        #print('内部last_hidden',last_hidden_list)
        #print('decode embed',embed)
        # run one step decoding
        #print('last_hidden_list[0]',last_hidden_list[0])
        hc_list = []

        h_t1, c_t1 = self.decoder_cell(embed, last_hidden_list[0])
        h_t2,c_t2 =self.decoder_cell2(h_t1,last_hidden_list[1])
        h_t3, c_t3 = self.decoder_cell3(h_t2, last_hidden_list[2])
        #h_t#, c_t# = self.decoder_cell#(h_t#-1, last_hidden_list[#-1])
        hc_list.append((h_t1,c_t1))
        hc_list.append((h_t2, c_t2))
        hc_list.append((h_t3, c_t3))
        #print('h_t1 1', h_t1)
        #print('h_t2 1', h_t2)
        #hc_list.append((h_t#-1, c_t#-1))
        #print('h_t3 2',h_t3)
        attn_scores = self.get_attn(h_t3, enc_outs, attn_mask)
        #print('decode attn_score',attn_scores)
        #attn_scores = self.get_attn(h_t#-1, enc_outs, attn_mask)
        context = attn_scores.matmul(enc_outs.transpose(0, 1))

        #context : [batch_size, 1, enc_out_dim]
        context = context.squeeze(1)
        # Luong eq.5.
        concat_out = torch.tanh(self.concat(
            torch.cat([context, h_t3], dim=1)
        ))

        logit = F.log_softmax(self.out(concat_out), dim=-1)
        return logit,hc_list


    def get_attn(self, dec_out, enc_outs, attn_mask):
        #implement attention mechanism
        keys = values = enc_outs
        query = dec_out.unsqueeze(0)

        #query: [1, batch_size, hidden_size]
        #enc_outs: [max_len, batch_size, hidden_size]
        #weights: [max_len, batch_size]
        weights = torch.sum(query * self.attn(keys), dim=2)
        weights = weights.transpose(0, 1)
        weights = weights.masked_fill(attn_mask==0, -1e18)
        weights = weights.unsqueeze(1)


        #另一种实现
        # values = enc_outs.transpose(0, 1) #[batch_size, max_len, hsize]
        # keys = self.attn(values).transpose(1, 2) #batch_size, hsize, max_len
        # query = dec_out.unsqueeze(1) #[batch_size, 1, hsize]
        # attn_scores = query.matmul(keys) #batch_size, 1, max_len
        # attn_scores = attn_scores.masked_fill(attn_mask==0, -1e18)
        # context = query.matmul(values) # batch_size, 1, hidden_size

        # return [batch_size, 1 max_len]
        return F.softmax(weights, dim=2)

#helper function
def len_mask(lens):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

#
# if __name__ == "__main__":
#     #test
#     model = Seq2SeqSum(300, 64, 128)
#     src = torch.randint(299, (32, 15)).long()
#     src_lengths =torch.randint(2, 14, (32,)).long()
#     lens = torch.LongTensor(list(reversed(sorted(src_lengths.tolist()))))
#     tgt = torch.randint(299, (32, 10)).long()
#     out = model(src, lens, tgt)
#     # print(out)
#