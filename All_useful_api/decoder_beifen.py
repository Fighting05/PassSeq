import time
import torch
from source.Seq2Seq_beifen import Seq2SeqSum
from source.utils import make_word2id
#############################################################################
def _pwd(pwd):
    temp = ''
    for i in range(len(pwd)):
        if i < len(pwd) - 1:
            temp+=pwd[i]
            temp+=' '
        else:
            temp+=pwd[i]
    return temp





def binarySearch(arr, l, r, x):
    print(arr)
    if x <= arr[0]:
        return 0
    if x >= arr[len(arr)-1]:
        return len(arr) - 1
    # 基本判断
    if r >= l:
        mid = int(l + (r - l) / 2)
        #print(r - l, mid)
        # 元素整好的中间位置
        if arr[mid] < x and x <= arr[mid + 1]:
            return mid

            # 元素小于中间位置的元素，只需要再比较左边的元素
        if arr[mid] > x:
            #print(mid)
            return binarySearch(arr, l, mid, x)
            # 元素大于中间位置的元素，只需要再比较右边的元素
        else:
            #print(mid)
            return binarySearch(arr, mid, r, x)

    else:
        # 不存在
        return -1

##############################################################################

# file_1 = open('show.txt', 'r')
# test = []
# while True:
#     line = file_1.readline()
#     if not line:
#         break
#     test.append(line[:-1])
# file_1.close()
def pwdfilter(pwdtmp,pwd):
    if len(pwd)!= len(pwdtmp):
        return False
    else:
        l = len(pwdtmp)
        for i in range(l):
            if pwdtmp[i]!='*' and pwdtmp[i]!=pwd[i]:
                return False
    return True


# def beam_search_df(model_path, word2id,path):
#     file = open(path,'w')
#     model = Seq2SeqSum(len(word2id), 128, 256, 3)
#     ckpt = torch.load(model_path)['state_dict']
#     model.load_state_dict(ckpt)
#     cnt = 0
#
#     for test_src in test:
#         hit=0
#         print('EXTENT:',test_src)
#         pwdtmp = test_src
#         print(pwdtmp)
#         cnt += 1
#         print('TEMPLATE:', cnt)
#         model.comb(test_src, word2id,10.0)
#         #model.pwd_probability('Pa$$w0rd',0,test_src, word2id)
#         #SENT, p = model.bs_decode(test_src, word2id, 1)
#         # for i in range(len(SENT)):
#         #     SENT[i] = SENT[i].replace(',', '\n')
#         #     SENT[i] = SENT[i].replace(' ', '')
#         #     #if len(SENT[i])>=6:
#         #     if float(p[i])>-15:
#         #         print(SENT[i],p[i])










import random
    # model.pwd_probability_normal('sunshinE',0,'* u * s h * n *',word2id)
    # model.montecarlo_sample(0,'* * * * * *',word2id)


def mask_attack_cal(f_in,f_out,thres):
    file_out = open(f_out, 'a')
    file_in = open(f_in, 'r')

    word2id = make_word2id("tmp/vocab.pkl", 95)
    model_path = 'output/PassSeqGuesser_model'#"output/ckpt-0.819260-12e-30000s"
    model = Seq2SeqSum(len(word2id), 128, 256, 3)
    ckpt = torch.load(model_path)['state_dict']
    model.load_state_dict(ckpt)

    while 1:
        line = file_in.readline()
        if line:
            A = line.strip('\n')
            if ' ' in A:
                continue
            pwd_tp = A
            pwd_tp_B = _pwd(pwd_tp)

            print('AA')
            model.mask_attack_A(pwd_tp_B, word2id, pwd_tp + '\n', 0, 0, 0, 0,thres,file_out)
            print('A')

        else:
            break








#两个beifen的代码才是可以用的
"""
重点就集中在Seq2Seq_beifen.py上
"""
if __name__ == "__main__":
    mask_attack_cal("PT.txt", 'result.txt')


    # start = time.time()
    # mask_attack_cal('F:\处理mask数据\\Wishbone_test.txt','Wishbone_test_result_ps.txt')
    # print('finished in {:.2f} seconds'.format(time.time() - start))




    # start = time.time()
    #
    # #beam_search_df(model_path, word2id,'store.txt')
    # print('finished in {:.2f} seconds'.format(time.time() - start))