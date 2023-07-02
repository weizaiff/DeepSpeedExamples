import json
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

def load_jsonl(data_path):
    data_list = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data_list.append(json.loads(line))
    return data_list 

'''
{"prompt": "新闻内容：昨天，北京市统计局、国家统计局北京调查总队联合发布北京人口调查报告，首次披露了环线人口分布情况。数据显示，人口分布呈现由二、三环内向四环外聚集的特点，五环外常住人口达1097.9万人，占全市的51%。65%外来人口住在四环到六环间 摘要：北京首次披露人口分布情况：超一半人口住五环外", 
"answers": [{"answer": "由于高房价造成居住地不断外延，而就业机会至今仍在四环内城区。造成潮汐式交通拥堵无解！上下班时间过长，占用越来越长的闲暇时间。使生活质量越来越低！", "score": 1}, 
{"answer": "城市空气污染、上班路上耗时、高额房租房价。。。", "score": 1}, 
{"answer": "人住五环外，就业还在五环内，钟摆效应让帝都交通不堪重负。//是的，可惜不是//", "score": 0}, 
{"answer": "北京五环内面积大致是700多平方公里，占全市面积16800平方公里的4%多点。", "score": 0}, 
{"answer": "领导身边福利好，希望全国人民都去领导身边享受福利哈", "score": 0},
 {"answer": "#数据新闻#我大帝都的人口分布。你住哪环？", "score": 0},
   {"answer": "五环外的服务设施配套完善了吗？", "score": 0}, 
   {"answer": "只是惊讶北京人已经这么多了…", "score": 0}], "prefix": "评论："}

   
'''
def convert_line(aline):
    '''
    目标格式： {"prompt": "Human: I have a question. Assistant:", 
    "chosen": "Good answer.", "rejected": "Bad answer."}.

    '''
    asample = (aline)
    res_list = []
    if len(asample['answers'])<1:return res_list
    for ans_i_s in range(len(asample['answers'])):
        for ans_i_e in range(ans_i_s+1, len(asample['answers'])):
            if asample['answers'][ans_i_s]['score']>asample['answers'][ans_i_e]['score']:
                # create sample
                tmp_s = {}
                tmp_s['prompt'] = asample['prompt']+' '+asample['prefix']
                tmp_s['chosen'] = asample['answers'][ans_i_s]['answer']
                tmp_s['rejected'] = asample['answers'][ans_i_e]['answer']
                res_list.append(tmp_s)
    return res_list
def main(is_beta, input_data_path, output_data_path):
    data_path = input_data_path
    
    data_list = load_jsonl(data_path)[:100000]
    if is_beta:
        data_list = data_list[:10000]
    print(data_list[0])
    n_p = 10
    #with Pool(n_p) as p:
        #feature_list = list(tqdm(p.map(convert_line, data_list),total=len(data_list)))
    feature_list = []
    with Pool(n_p) as pool:
        with tqdm(total=len(data_list)) as pbar:
            for result in pool.imap_unordered(convert_line, data_list):
                # 处理每个 result
                feature_list.append(result)
                pbar.update(1)
    #展开
    final_feature_list = []
    for ifea in feature_list:
        for i_ifea in ifea:
            final_feature_list.append(i_ifea)
    if False:
        json_str = ''
        for in_fea in tqdm(final_feature_list):
            json_str+=(json.dumps(in_fea, ensure_ascii=False).encode('utf8'))
            json_str+='\n'
        with open(output_data_path, 'w') as f:
            print('write...')
            f.write(json_str)
    #with open(output_data_path, 'w') as outfile:
    #    for entry in final_feature_list:
    #        json.dump(entry, outfile)
    #        outfile.write('\n')
    prompt=[]
    chosen=[]
    rejected=[]
    for entry in final_feature_list:
        p_ = entry['prompt']
        #生成MOSS数据格式的RW数据
        p_=f'''我是人工智能助手Happy! 能够帮助大家解决通识问题，我的目标是方便人类的生活，不创造或者生成任何有悖法律的回答，敬请提问！[Human]: {p_}<eoh> <|Inner Thoughts|>: None<eot>
<|Commands|>: None<eoc>
<|Results|>: None<eor>
[MOSS]:
        '''
        prompt.append(p_)
        chosen.append(entry['chosen'])
        rejected.append(entry['rejected'])
    dev_num =-100000
    df_train=pd.DataFrame({'prompt':prompt[:dev_num], 'chosen':chosen[:dev_num],'rejected':rejected[:dev_num]})
    df_train.to_csv(output_data_path+'train.csv', index=False)

    df_train=pd.DataFrame({'prompt':prompt[dev_num:], 'chosen':chosen[dev_num:],'rejected':rejected[dev_num:]})
    df_train.to_csv(output_data_path+'eval.csv', index=False)


if __name__=='__main__':
    is_beta = False
    #input_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/train_data_external_v1.jsonl'
    #output_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/train.json'
    #input_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/dev_data_external_v1.jsonl'
    #output_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/eval.json'

    #main(is_beta, input_data_path, output_data_path)

    input_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/train_data_external_v1.jsonl'
    output_data_path = '/mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus/'

    main(is_beta, input_data_path, output_data_path)

