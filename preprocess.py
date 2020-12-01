import random
import configparser
import os
import pandas as pd
import random
import fnmatch

config = configparser.ConfigParser()
config.read('../config.ini')


def get_Top_user():
    patten = '*-weibo.csv'
    weibo_file_list = fnmatch.filter(os.listdir('../dataset'), patten)
    print(weibo_file_list)
    for eachFile in weibo_file_list:
        df_temp = pd.read_csv('../dataset/' + eachFile, encoding='utf-8')
        df_temp.sort_values(by=['点赞数', '评论数', '转发数'], ascending=[False, False, False])
        print("len of dataset :" + str(df_temp.shape[0]))
        top_percent = int(df_temp.shape[0] * 0.2)
        df_top = df_temp.head(top_percent)['user_id']
        df_top.to_frame().drop_duplicates(subset=['user_id'], keep='first', inplace=True)
        print("top 20% is :" + str(df_top.shape[0]))
        df_top.to_csv("../dataset/top-userid.csv", header=True)


def split_expert():
    patten = '*-weibo.csv'
    weibo_file_list = fnmatch.filter(os.listdir('../dataset'), patten)
    print(weibo_file_list)
    for eachFile in weibo_file_list:
        print("current file : " + eachFile)
        type = eachFile.replace('-weibo.csv', '')
        normalfile = '../dataset/ClassificationResult-' + type + '-normal' + '.csv'
        expertfile = '../dataset/ClassificationResult-' + type + '-expert' + '.csv'
        df_normal = pd.read_csv(normalfile, encoding='utf-8')
        delet_list = df_normal[df_normal['user_id'].str.len() != 10].index.tolist()
        df_normal.drop(index=delet_list, inplace=True)

        # delete duplication
        df_normal = df_normal.drop_duplicates(subset='Text', keep='last')
        # transform user_id from sting to int
        df_normal['user_id'] = df_normal['user_id'].apply(int)
        df_user = pd.read_csv('../dataset/top-userid.csv', encoding='utf-8')
        df_user['user_id'] = df_user['user_id'].apply(int)
        ids = df_user['user_id'].values.tolist()
        df_expert = df_normal[df_normal["user_id"].isin(ids)]
        df_expert.to_csv(expertfile)
    pass


def preprocess():
    # file name style: 上证指数-normal-weibo.csv
    #                  上证指数-expert-weibo.csv
    # pathdt = '../dataset/weibo_senti_100k.csv'
    # df_train = pd.read_csv(pathdt,encoding = 'utf-8')#循环处理路径下所有的微博文件
    patten = '*-weibo.csv'
    weibo_file_list = fnmatch.filter(os.listdir('../dataset'), patten)
    print(weibo_file_list)
    for eachFile in weibo_file_list:
        df_test = pd.read_csv('../dataset/' + eachFile, encoding='utf-8')
        testfile = 'sentiment.test' + '.' + eachFile.replace('-weibo.csv', '')
        print('name of test file is  ' + testfile)
        with open(os.path.join(config['path']['DATA_SET'], testfile), 'w', encoding='utf-8') as sd:
            for index, item in df_test.iterrows():
                text = str(item['text']) + '\t' + str(item['user_id']) + '\t' + str(item['time']) + '\n'
                #  + str(item['']) + '\t' + str(item['']) + '\t' + str(item['']) + '\n'
                sd.write(text)


    # with open(os.path.join(config['path']['DATA_SET'], 'sentiment.train'), 'w',encoding = 'utf-8') as st:
    #    for index,item in df_train.iterrows():
    #        text = item['review'] + '\t' + str(item['label']) +'\n'
    #        st.write(text)


if __name__ == "__main__":
    #get_Top_user()
    #preprocess()
    split_expert()
