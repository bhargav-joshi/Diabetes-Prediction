import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dfx=pd.read_csv("Diabetes_XTrain.csv")
dfy=pd.read_csv("Diabetes_YTrain.csv")
dfTest=pd.read_csv("Diabetes_Xtest.csv")
X=dfx.values
Y=dfy.values
Y=Y.reshape((-1,))
def dist(X1,X2):
    return np.sqrt((sum(X1-X2)**2))
def knn(X,Y,querypoint,k=25):
    values=[]
    for i in range(X.shape[0]):
        val=dist(querypoint,X[i])
        values.append((val,Y[i]))
    values=sorted(values)
    values=values[:k]
    values=np.array(values)
    new_values=np.unique(values[:,1],return_counts=True)
    max_freq=np.argmax(new_values[1])
    pred=new_values[0][max_freq]
    return(pred)
Test=dfTest.values
lis=[]
for i in range(192):
    lis.append(knn(X,Y,Test[i]))
    
diabetes={
   "Outcome":np.asarray(lis)
}

df=pd.DataFrame(diabetes,dtype='int32')
df.to_csv("Solution.csv",index=False)
