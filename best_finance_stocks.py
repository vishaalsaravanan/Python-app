import requests
import pandas as pd 
from yahoo_fin import stock_info as si 
from pandas_datareader import DataReader
import numpy as np
from flask import Flask,jsonify
from flask_jsonpify import jsonpify

app=Flask(__name__)

"""
TODO 
1.Check for baahubalii

"""

@app.route('/')
def landing():
    tickers = si.tickers_sp500()
    tickers=tickers[0:5] #for testing 
    recommendations = []

    for ticker in tickers:
        lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
        rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
                  'modules=upgradeDowngradeHistory,recommendationTrend,' \
                  'financialData,earningsHistory,earningsTrend,industryTrend&' \
                  'corsDomain=finance.yahoo.com'
                  
        url =  lhs_url + ticker + rhs_url
        r = requests.get(url)
        if not r.ok:
            recommendation = 6
        try:
            result = r.json()['quoteSummary']['result'][0]
            recommendation =result['financialData']['recommendationMean']['fmt']
        except:
            recommendation = 6
        
        recommendations.append(recommendation)
    dataframe = pd.DataFrame(list(zip(tickers, recommendations)), columns =['Company', 'Recommendations']) 
    dataframe = dataframe.set_index('Company')
    result={}
    dataframe.insert(0,"Tickers",tickers)
    for index,row in dataframe.iterrows():
        result[index]=dict(row)
    dataframe.to_csv('recommendations.csv')

    return jsonify(result)
    


if __name__ =='__main__':

    
    
    app.run()
    
