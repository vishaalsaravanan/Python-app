import pandas as pd
import numpy as np

company_df = pd.read_csv('companylist.csv')

rec_df=pd.read_csv('recommendations.csv')

ctickers=company_df.iloc[:,0].values
rtickers=rec_df.iloc[:,0].values

final_arr=[]
outer_index=0
for each_ticker in rtickers:
    inner_index=0
    for stock_names in ctickers:
        
        if each_ticker==stock_names:
            names=company_df.iloc[inner_index,1]
            rs=rec_df.iloc[outer_index,1]
            new_entry={'Stock Ticker':each_ticker,'Stock Names':names,'Recommendation Score':rs}
            final_arr.append(new_entry.copy())
        inner_index+=1
    outer_index+=1

df=pd.DataFrame(final_arr)
df.to_csv('Dashboard_Rec.csv')