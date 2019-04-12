import csv
import pandas as pd
import sys
import os
import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import argparse
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)
        
def Descargar_Datos(consumer_token,consumer_secret,access_token,access_token_secret):

    auth = tweepy.OAuthHandler(consumer_token,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    import json

    #archivos con los ids de los doferentes datasets
    names = ['Waseem_Dataset','Data_new']
    archivos_ids = ['.\Data\Waseem_IDS.csv', '.\Data\Data_new_IDS.csv ']
    total = 0
    for current_file in range(len(archivos_ids)):
        c = 0
        print('Downloading ' + str(current_file + 1) + ' of 2 datasets ...')
        data =  pd.read_csv(archivos_ids[current_file],'r',delimiter = ',',encoding = 'utf-8')
        str_json = []
        for j in data.values:
            total += 1
            if total % 100 ==0:
                print(total)
            try:
                tweet = api.get_status(j[0])._json
                str_json.append({"id":tweet['id'],"name": tweet['user']['name'], "text":tweet['text'], "label": j[1]})
                c += 1
                if c % 100 ==0:
                    print('recuperados')
                    print(c)
            except: 
                pass
        print(str(c) + ' tweets downloaded')
        save_object(str_json, '.\Data\' + names[current_file] + '.pkl')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descargar datos usando el api de twitter')
    parser.add_argument('-ct', '--consumer_token', required=True)
    parser.add_argument('-cs', '--consumer_secret', required=True)
    parser.add_argument('-at', '--access_token', required=True)
    parser.add_argument('-ats', '--access_token_secret', required=True)

    consumer_token = parser.parse_args().consumer_token
    consumer_secret = parser.parse_args().consumer_secret
    access_token = parser.parse_args().access_token
    access_token_secret = parser.parse_args().access_token_secret
    
    Descargar_Datos(consumer_token,consumer_secret,access_token,access_token_secret)