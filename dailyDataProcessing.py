import pandas as pd
from pandas.tseries.offsets import Day, MonthEnd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import fnmatch
import os


def prepare_stock_info(type):
    currentFile = '../dataset/' + type + '-price.csv'
    stock_file = pd.read_csv(currentFile, encoding='utf-8')
    stock_file.drop(columns=["Close", "High", "Low", "Amount", "Rate"], inplace=True)
    stock_file["Date"] = pd.to_datetime(
        stock_file["Date"].apply(lambda x: x.replace('年', '/').replace('月', '/').replace('日', '')))
    if '恒生' in type:
        stock_file["Open"] = stock_file["Open"].str.replace(',', '').astype(float)
    return stock_file


def compute_daily_sentimentValue(type, user_group):
    currentFile = '../dataset/ClassificationResult-' + type + '-' + user_group + '.csv'
    df = pd.read_csv(currentFile)
    df.drop(columns=['ID', 'Text'], inplace=True)
    daily_sentiment = df.groupby('Date').mean().reset_index()
    delet_list = daily_sentiment[daily_sentiment['Date'].str.len() > 10].index.tolist()
    daily_sentiment.drop(index=delet_list, inplace=True)
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    # daily_sentiment.to_csv('../dataset/daily_sentiment_mean.csv', index=False)

    return daily_sentiment


def compute_T_value(daily_sentiment_dataframe, stock_dataframe, type , user_group):
    merged = pd.merge(daily_sentiment_dataframe, stock_dataframe, how='inner', on=['Date'])
    # spearman = merged.corr(method='spearman')
    # print("spearman: ")
    # print(spearman)
    senti = merged['Expected'].values.tolist()
    price = merged['Open'].values.tolist()
    date = merged['Date'].tolist()
    biggest_T = 0
    biggest_P = 0.00
    p_list=[]
    for t in [3, 5, 7, 9, 12, 15, 20, 25, 30]:
        senti_temp = senti[t:]
        price_temp = price[:len(price) - t]
        pearson = stats.pearsonr(senti_temp, price_temp)
        #kruskal = stats.kruskal(senti_temp,price_temp)
        # spearman = stats.spearmanr(senti_temp, price_temp)
        #print("while T == " + str(t) + " pearson is : " + str(abs(pearson[0])) + " , p value is : " + str(pearson[1]))
        p_list.append(abs(pearson[0]))
        #print("while T == " + str(t) + " kruskal is : " + str(abs(kruskal[0])) + " , p value is : " + str(kruskal[1]))
        if abs(pearson[0]) > abs(biggest_P):
            biggest_P = pearson[0]
            biggest_T = t
    print(p_list)
    dataframe = pd.DataFrame({'date': date[:len(date) - t], 'sentiment': senti[t:], 'price': price[:len(price) - t]})
    sentiment_file_after = '../dataset/sentimentDaily-' + type + user_group + '.csv'
    dataframe.to_csv(sentiment_file_after, index=False, sep=',')
    return biggest_T, biggest_P


def main():
    patten = '*-weibo.csv'
    weibo_file_list = fnmatch.filter(os.listdir('../dataset'), patten)
    for eachFile in weibo_file_list:
        for user_group in ['normal', 'expert']:
            type = eachFile.replace('-weibo.csv', '')
            spd = compute_daily_sentimentValue(type, user_group)
            ppd = prepare_stock_info(type)
            biggest_T, biggest_P = compute_T_value(spd, ppd, type,user_group)
            print('for type: ' + type + user_group + ' ,the result is: ' + 'T = ' + str(biggest_T) + ', pearson = ' + str(biggest_P))


if __name__ == "__main__":
    main()
