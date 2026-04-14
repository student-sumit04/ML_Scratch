#csv
#numpy,2x
#pandas

#missing data, header , dtype



import csv
import numpy as np
import pandas as pd


FILE_NAME="spambase.data"
##with open(FILE_NAME,'r') as f:
  #  data=list(csv.reader(f,delimiter=","))
#data=np.array(data)

#by numpy

#data=np.loadtxt(FILE_NAME,delimiter=",")
data=np.genfromtxt(FILE_NAME,delimiter=",",dtype=np.float32,skip_header=1,missing_values="Hello",filling_values=0)

print(data.shape,type(data[0][0])) 


n_sample,n_feature=data.shape
n_feature-=1
X=data[:,0:n_feature]
y=data[:,n_feature]
print(X.shape,y.shape)
print(X[0,0:5])


#by pandas

df=pd.read_csv(FILE_NAME,delimiter=",",header=None,dtype=np.float32,skiprows=1,na_values=["Hello"])#here skip rows in pandas and  skip_header in numpy 
df=df.fillna(0)
data=df.to_numpy()
#data=np.asarray(data,dtype=np.float32)

n_sample,n_feature=data.shape
n_feature-=1
X=data[:,0:n_feature]
y=data[:,n_feature]
print(X.shape,y.shape)
print(X[0,0:5])




    