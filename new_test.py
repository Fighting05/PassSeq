import os
import time
import random
import All_useful_api.make_vocab_second
import All_useful_api.train_seq2seqsum
import All_useful_api.decoder_beifen

logFunction = lambda s: print(s)

def check(s):
    if len(s) < 4 or len(s) > 31:
        return False
    for c in s:
        if (ord(c) < 32 or ord(c) > 127) and (c!='*' and c!=' ') :
            return False
    return True







class PassSeqGuesser:
    def __init__(self, paras):
        self.trainPath = paras['trainPath']
        self.outfile = paras['outfile']
        self.threshold = paras['threshold']

        # self.gram = int(paras['gram'])
        # self.guessNum = int(paras['guessNum'])

        self.tmpPath = 'tmp/tmp.txt'

    def s2smakevocab(self, mask_pro,dataf):
        file = open(dataf, 'r')
        f = ''
        if mask_pro == 0.5:
            f = 'tmp/input.txt'
        if mask_pro == 1.0:
            f = 'tmp/output.txt'
        fi = open(f, 'w')
        while 1:
            line = file.readline()
            if line:
                line = line.replace('\n', '')
                mask_line = ''
                for i in range(len(line)):
                    morno = random.uniform(0, 1)
                    if i < len(line) - 1:
                        if morno >= mask_pro:
                            mask_line += '*'
                            mask_line += ' '
                        else:
                            mask_line += line[i]
                            mask_line += ' '
                    if i == len(line) - 1:
                        if morno >= mask_pro:
                            mask_line += '*'
                        else:
                            mask_line += line[i]
                fi.write(mask_line + '\n')
            else:
                break

    def s2strainsetgene(self,dataf):
        def maskprocess(line, mask_pro):
            mask_line = ''
            for i in range(len(line)):
                morno = random.uniform(0, 1)
                if i < len(line) - 1:
                    if morno >= mask_pro:
                        mask_line += '*'
                        mask_line += ' '
                    else:
                        mask_line += line[i]
                        mask_line += ' '
                if i == len(line) - 1:
                    if morno >= mask_pro:
                        mask_line += '*'
                    else:
                        mask_line += line[i]
            return mask_line

        def blanprocess(line):
            mask_line = ''
            for i in range(len(line)):
                if i < len(line) - 1:
                    mask_line += line[i]
                    mask_line += ' '
                if i == len(line) - 1:
                    mask_line += line[i]
            return mask_line


        file_t_a = open('tmp/train_article.txt', 'w')
        file_t_t = open('tmp/train_title.txt', 'w')
        file_v_a = open('tmp/val_article.txt', 'w')
        file_v_t = open('tmp/val_title.txt', 'w')
        file = open(dataf, 'r',
                       encoding='utf-8', errors='ignore')
        while True:
            line = file.readline()
            if not line:
                break
            line = line.replace(' ', '').replace('\n', '')

            torv = random.uniform(0,1)
            if torv>=0.2:
                blanline = blanprocess(line)
                file_t_t.write(blanline+'\n')
                mask_line = maskprocess(line,0.5)
                file_t_a.write(mask_line+'\n')
            else:
                blanline= blanprocess(line)
                file_v_t.write(blanline+'\n')
                mask_line = maskprocess(line, 0.5)
                file_v_a.write(mask_line + '\n')
        file.close()
        file_v_t.close()
        file_v_a.close()
        file_t_a.close()
        file_t_t.close()


    def filter(self):


        msg = "Preprocessing data...\n"
        logFunction(msg)

        fp = open(self.tmpPath, 'w', encoding='utf-8')
        #count=0
        with open(self.trainPath, 'r', encoding='utf-8') as ifp:
            line = ifp.readline()

            while line:
                line = line.strip('\n')

                if check(line):
                    fp.write(line + '\n')
                    # count+=1
                    # if count==10000:
                    #     break

                line = ifp.readline()

        fp.close()


        msg = "字典正在制作\n"
        logFunction(msg)
        self.s2smakevocab(0.5,self.tmpPath)
        self.s2smakevocab(1.0, self.tmpPath)
        SRC_PATH = "tmp/input.txt"
        TARGET_PATH = "tmp/output.txt"
        VOCAB_PATH = 'tmp/vocab.pkl'
        All_useful_api.make_vocab_second.make_vocab(SRC_PATH, TARGET_PATH, VOCAB_PATH)
        msg = "字典制作完成\n"
        logFunction(msg)

        msg = "训练集和验证集正在制作\n"
        logFunction(msg)
        self.s2strainsetgene(self.tmpPath)
        msg = "训练集和验证集制作完成\n"
        logFunction(msg)


    def guess(self):
        # msg = "Training...\n"
        # logFunction(msg)
        # All_useful_api.train_seq2seqsum.main(128,256)
        # msg = "Finish Training...\n"
        # logFunction(msg)

        msg = "Start Guessing...\n"
        logFunction(msg)
        All_useful_api.decoder_beifen.mask_attack_cal("tmp/PT.txt",'outfile/myspace-guess.txt',self.threshold)
        msg = "Finish Guessing...\n"
        logFunction(msg)








    def solve(self):
        #self.filter()
        self.guess()

        #msg = "Generation End."
        #logFunction(msg)


if __name__ == '__main__':
    paras = {
        'trainPath': "data/myspace.txt",
        'outfile': "myspace-guess.txt",
        'threshold': 5e-6
    }

    PG = PassSeqGuesser(paras)

    PG.solve()

