import csv
import pandas as pd
import sys
import os
import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener

def Descargar_Datos(consumer_token,consumer_secret,access_token,access_token_secret)

    auth = tweepy.OAuthHandler(consumer_token,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    import json

    #archivos con los ids de los doferentes datasets
    archivos_ids = ['.\Data\Waseem_IDS.csv']
    archivos_datos = ['.\Data\Waseem_Dataset.json']

    c = 0
    for file_number in range(len(archivos_ids)):
        data =  pd.read_csv(archivos_ids[file_number],'r',delimiter = ',',encoding = 'utf-8')
        output_file = open(archivos_datos[file_number], 'a')              
        str_json = []
        for j in data.values:
            try:
                tweet = api.get_status(j[0])._json
                str_json.append({"id":tweet['id'],"name": tweet['user']['name'], "text":tweet['text'], "label": j[1]})

            except: 
                pass
        save_object(str_json,'Waseem_Dataset.pkl')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descargar datos usando el api de twitter')
    parser.add_argument('-ct', '--consumer_token')
    parser.add_argument('-cs', '--consumer_secret')
    parser.add_argument('-at', '--access_token')
    parser.add_argument('-ats', '--access_token_secret')

    consumer_token = parser.parse_args().consumer_token
    consumer_secret = parser.parse_args().consumer_secret
    access_token = parser.parse_args().access_token
    access_token_secret = parser.parse_args().access_token_secret
    
    Descargar_Datos(consumer_token,consumer_secret,access_token,access_token_secret)