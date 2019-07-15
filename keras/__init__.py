import pandas as pd
import numpy as np
path = 'C:\\Users\TangGuoping\Desktop\hotitem-top10000-datepart=20190630.csv'
data = pd.read_csv(path, encoding ='latin1')
data.sample(3)
gg=data.groupby('itemId').count()
gg1=gg.index.tolist()
print(gg1)