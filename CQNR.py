def CQNR():
    import tkinter as tk

    class SampleApp(tk.Tk):
      def __init__(top):
        tk.Tk.__init__(top)
        top.geometry('750x300+500+300')
        top.title('CQNR')
        top.configure(background='plum1')
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=0,column=3)
        #top.newline = tk.Label(top, text="", bd =5).grid(row=1,column=3)
        
        top.caption = tk.Label(top, text="Please fill the following fields", bd =5, bg='plum1').grid(row=2,column=1)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=3,column=1)
        top.label1 = tk.Label(top, text="Imbalanced class data file location (csv seperate by ',' ; row:samples, columns:features)", bd =5, bg='plum1').grid(row=6,column=1)
        top.label2 = tk.Label(top, text="Output location for the balanced data file", bd =5, bg='plum1').grid(row=8,column=1)
        
        top.entry1 = tk.Entry(top, bd =3, width=40)
        top.entry2 = tk.Entry(top, bd =3, width=40)
        
        top.button = tk.Button(top, text="Oversample!", command=top.on_button, padx=2, pady=2, width=10, bg="bisque2")
        top.entry1.grid(row=6, column=2)
        top.entry2.grid(row=8, column=2)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=13,column=1)
        top.button.grid(row=16, column=2)
    
        

      def on_button(top):
        x1=top.entry1.get()
        x2=top.entry2.get()
        top.destroy()
        daviesbouldin(x1,x2)
        

    app = SampleApp()
    
    app.mainloop()



def daviesbouldin(input, output):
    #mainly used for plotting the test cases of CQNR-OS
    from sklearn import svm
    import math
    import csv
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from random import shuffle
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    import numpy as np
    from scipy.cluster.hierarchy import cophenet
    from sklearn.cluster import MeanShift
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    import pandas
    import decimal
    import random
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    f1=random.seed()
    from sklearn.model_selection import cross_val_score
    from imblearn.over_sampling import SMOTE


    #single file
    #dataframe = pandas.read_csv("G:/rishika/3_work_complex_network_effector/combination/all8features(3vs4vs6vsnon).csv", header=None, sep=',')
    dataframe = pandas.read_csv(input, header=None, sep=',')

    dataset = dataframe.values
    k1=dataset.shape
    print(k1[1])
    X = dataset[:,0:k1[1]-1].astype(float)
    Y = dataset[:,k1[1]-1].astype(int)
    #print(Y)
    X3=[]
    X4=[]
    #print(type(X[0]))

    #more than 2 class
    
    un=np.unique(Y)
    #print(un)
    max_class=0
    max_count=0
    for i in range(len(un)):
       count=0
       for j in range(len(Y)):
          if Y[j]==un[i]:
             count=count+1
       if count>max_count:
          max_count=count
          max_class=un[i]
    print(max_class)
    X4=[]
    X3=[]
    for i in range(len(Y)):
       if Y[i]==max_class:
           X4.append(X[i])
    n=X.shape
    print(n)
    X_syn=[]
    Y_syn=[]
    
    for i in range(len(un)):
       
       for j in range(len(Y)):
          if Y[j]!= max_class and Y[j]==un[i]:
             X3.append(X[j])
          
       if len(X3)>len(X4):
         smaller=X4
         larger=X3
       else:
         smaller=X3
         larger=X4
       
       if not X3:
          continue
         
       new_sample=syntheticsample(smaller,larger,n[1])
       
       
       print('\n -------------------------')
       size=n[1]
       
       S1 = [[0 for x in range(size)] for y in range(len(smaller))]
       for i1 in range(len(smaller)):
          f=smaller[i1].tolist()
          S1[i1][:]=f[:]
          
       S2 = [[0 for x in range(size)] for y in range(len(new_sample))]
       for i1 in range(len(new_sample)):
          f=new_sample[i1].tolist()
          S2[i1][:]=f[:]
          
       #combining the datas
       for i1 in range(len(S1)):
            X_syn.append(S1[i1][:])
            Y_syn.append(i+1)
       
       for i1 in range(len(S2)):
            X_syn.append(S2[i1][:])
            Y_syn.append(i+1)
            i1=i1+1
       print('smaller:',len(smaller))
       print('larger:',len(larger))
       print('new sample:',len(new_sample))
       X3=[]
       smaller=[]
       #larger=[]
       new_sample=[]
    #print(len(X_syn))  
    size=n[1]
    S3 = [[0 for x in range(size)] for y in range(len(larger))]
    for i1 in range(len(larger)):
          f=larger[i1].tolist()
          S3[i1][:]=f[:]
    for i1 in range(len(S3)):
            X_syn.append(S3[i1][:])
            Y_syn.append(max_class)      
    
    S4 = [[0 for x in range(size+1)] for y in range(len(X_syn))]
    for i1 in range(len(X_syn)):
          S4[i1][0:size]=X_syn[i1]
          S4[i1][size]=Y_syn[i1]
    
    #print(S4[0], S3[0])
    #file=open('davies.csv','w')
    file=open(output,'w')
    
    with file:
       writer=csv.writer(file)
       writer.writerows(S4)
    print('writing complete')   


def syntheticsample(smaller, larger, dim):
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas
    import math
    import random
    from scipy.spatial import distance
    f1=random.seed()
    
    min_dbi=1000
    max_c=0
    num=2
    while num < 20:
      Z=KMeans(n_clusters=num)
      X=Z.fit(smaller)
      P=X.labels_
      #print(P)
      #print(len(smaller))
      
      center = [[0 for x in range(dim)] for y in range(num)]
      for i in range(num):
       max_diam=0

      #validity index
      sum_1 = [[0 for x in range(dim)] for y in range(1)]

      #cluster center
      for i in range(num):
        count=0  
        for j in range(len(P)):
           if P[j]==i:
              v1=[]
              v1=smaller[j][:]
              for h in range(dim):
                  sum_1[0][h]=sum_1[0][h]+v1[h]
              count=count+1
        for h in range(dim):
           center[i][h]=sum_1[0][h]/count
      #print(center)
      center=X.cluster_centers_
      #print(X.cluster_centers_)
      #average distance of each point from cluster center
      sum_2 = [[0 for x in range(dim)] for y in range(1)]
      #sum_2=0     
      S = [[0 for x in range(1)] for y in range(num)]
      for i in range(num):
        count=0
        X1=0
        for j in range(len(P)):
           if P[j]==i:
              v1=[]
              v1=smaller[j][:]
              for h in range(dim):
                  sum_2[0][h]=sum_2[0][h]+(center[i][h]-v1[h])**2
                  #sum_2=sum_2+distance.euclidean(center[i],v1)
              count=count+1
              
        for h in range(dim):
               #print(type(sum_2[0][h]), sum_2[0][h])
         X1=X1+sum_2[0][h]
        X1=X1/count
        S[i]=X1**(0.5)

      #Distance between each centroid
      M = [[0 for x in range(num)] for y in range(num)]
      for i in range(num):
          
        for j in range(num):
          t=0  
          if i != j:
           for g in range(dim):
             t=t+(center[i][g]-center[j][g])**2
             #t=t+distance.euclidean(center[i],center[j])
           M[i][j]=t**0.5
          
      #finding R(i,j)
      R = [[0 for x in range(num)] for y in range(num)]     
      l1=0
      for i in range(num):
        for j in range(num):
          if i !=j:
           R[i][j]=(S[i]+S[j])/M[i][j]
                
      
      #Finding Di
      D = [[0 for x in range(1)] for y in range(num)]     
      max=0
      for i in range(num):
        for j in range(num):
          if i!=j:  
           if R[i][j]>max:
            max=R[i][j]
        D[i]=max
        max=0

      #finding db
      dbi=0  
      for i in range(num):
          dbi=dbi+D[i]
      dbi=dbi/num    
  
      print(num,dbi)
      if min_dbi>dbi:
         min_dbi=dbi
         max_c=num
         
      num=num+1
    
    print(max_c)
    #final clustering
    Z=KMeans(n_clusters=max_c)
    X=Z.fit(smaller)
    P=X.labels_
    #print(P)
    d=len(larger)-len(smaller)
    #print(d)
    i=0
    
    freq=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    num_old=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    #print(np.unique(P))
    while i<len(np.unique(P)):
       count=0
       for j in range(len(P)):
          if P[j]==i:
             count=count+1
       #print(count)
       num_old[i]=count
       freq[i]=count/len(smaller)*100
       #p=random.randint(1,k-1)
       i=i+1
    #print(freq)
    #print(num_old)
    new=[[0 for x in range(1)] for y in range(len(np.unique(P)))]
    new_sample=[[0 for x in range(2)] for y in range(d)]
    new_sample=[]
    i=0
    while i<len(freq):
       new[i]=math.floor((freq[i]*d)/100+0.5)
       i=i+1
    #print(new)

    #new sets
    i=0
    #print(new)
    while i<(len(np.unique(P))):
       g=0
       while g<new[i]:
         i1=random.randint(0,num_old[i]-1)
         i2=random.randint(0,num_old[i]-1)
         #print(i1,i2)
         k=-1
         for j in range(len(P)):
            if P[j]==i:
               k=k+1
               if k==i1:
                  val1=smaller[j][:]
                  v1=j
               if k==i2:
                  val2=smaller[j][:]
                  v2=j
               #print(j)
         w1=random.uniform(0,1)
         w2=1-w1
         
         f=(w1*val1)+(w2*val2)
         #print(v1,v2)
         #print(f.astype(int))
         #int when integer dataset else no int
         #b=f.astype(int)
         new_sample.append(f)
         g=g+1
       i=i+1
       

    n=np.shape(smaller)

    return new_sample
