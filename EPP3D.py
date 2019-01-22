def EPP3D():
    import tkinter as tk
    import os
    from tkinter import filedialog
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror

    
    
    
    class SampleApp(tk.Tk):
      def __init__(top):
        tk.Tk.__init__(top)
        top.geometry('650x300+500+300')
        top.title('EPP3D')
        top.configure(background='plum1')
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=0,column=3)
        #top.newline = tk.Label(top, text="", bd =5).grid(row=1,column=3)
        
        top.caption = tk.Label(top, text="Please insert the file names for executing PyPredT6", bd =5, bg='plum1').grid(row=2,column=1)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=3,column=1)
        top.label1 = tk.Label(top, text="Training dataset (csv format)", bd =5, bg='plum1').grid(row=6,column=1)
        #top.button = tk.Button(top, text="Browse", command=top.load_file, padx=2, pady=2, width=10, bg="bisque2").grid(row=6, column=4)
        #x=os.path.abspath(fname)
        
        top.label2 = tk.Label(top, text="PDB file of protein to be predicted", bd =5, bg='plum1').grid(row=8,column=1)
##        top.label3 = tk.Label(top, text="Effector feature file", bd =5, bg='plum1').grid(row=10,column=1)
##        top.label4 = tk.Label(top, text="Non-effector feature file", bd =5, bg='plum1').grid(row=12,column=1)
        
        top.entry1 = tk.Entry(top, bd =3, width=40)
        top.entry2 = tk.Entry(top, bd =3, width=40)
##        top.entry3 = tk.Entry(top, bd =3, width=40)
##        top.entry4 = tk.Entry(top, bd =3, width=40)
        
        top.button = tk.Button(top, text="Predict!", command=top.on_button, padx=2, pady=2, width=10, bg="bisque2")
        top.entry1.grid(row=6, column=2)
        top.entry2.grid(row=8, column=2)
##        top.entry3.grid(row=10, column=2)
        
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=13,column=1)
        top.button.grid(row=16, column=2)
    
        
##      def load_file(top):
##        filename = askopenfilename(filetypes=(("PDB files", "*.pdb"),
##                                           ("All files", "*.*") ))
##        label=filename
##        print(label)
##        if fname:
##            try:
##                print("""here it comes: self.settings["template"].set(fname)""")
##            except:                     # <- naked except is a bad idea
##                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
##            return
      def on_button(top):
        x1=top.entry1.get()
        x2=top.entry2.get()
##        x3=top.entry3.get()
##        x4=top.entry4.get()
        top.destroy()
        prediction(x1,x2)
        
##    print(label)
    app = SampleApp()
    
    
    app.mainloop()


def prediction(training, testing):
    import random
    import pandas
    import numpy as np
    import csv
    from sklearn import svm
    # importing necessary libraries
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from random import shuffle
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    f=random.seed()
    from sklearn.metrics import accuracy_score
    import numpy as np
    np.random.seed(123)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    import keras.utils
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from imblearn.over_sampling import SMOTE, ADASYN
    from collections import Counter
    from sklearn.ensemble import ExtraTreesClassifier
    import warnings
    from sklearn.feature_selection import RFE
    from sklearn.preprocessing import label_binarize
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore")

    #extract the features from pdb files of the proteins to be classified
    feature=pdbextract(testing)
    print(feature)
    #read dataset for training
    #dataframe = pandas.read_csv('H:/rishika/3_work_complex_network_effector/combination/davies(allvsnon).csv', header=None, sep=',')
    dataframe = pandas.read_csv(training, header=None, sep=',')
    dataset = dataframe.values
    X = dataset[:,0:8].astype(float)
    Y = dataset[:,8].astype(int)
    classnum=5

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=f)


    if classnum==5:
        y1 = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
        y2 = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    print('Training classifiers...')
    print('Training Multi layer perceptron...')
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(8,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Add an output layer 
    model.add(Dense(classnum, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.fit(X_train, y1, epochs=1000, batch_size=25, verbose=0)
    score = model.evaluate(X_test, y2,verbose=0)
    #print('ANN', score)
    y_predANN1=model.predict(X_test)


    newmodel = Sequential()
    newmodel.add(Dense(9, activation='relu', input_shape=(8,)))
    newmodel.add(Dense(10, activation='relu'))
    newmodel.add(Dense(30, activation='relu'))
    newmodel.add(Dense(20, activation='relu'))
    newmodel.add(Dense(50, activation='relu'))
    newmodel.add(Dense(20, activation='relu'))
    newmodel.add(Dense(10, activation='relu'))
    # Add an output layer 
    newmodel.add(Dense(classnum, activation='sigmoid'))
    newmodel.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    old_weights = model.get_weights()
    newmodel.set_weights(old_weights)
    feature1=np.asarray(feature)
    ANN=newmodel.predict(feature1, batch_size=1)

    print("Training Support Vector Machine...") 
    clf1 = svm.SVC(decision_function_shape='ovo', kernel='rbf', max_iter=10000)
    clf1.fit(X_train, y_train)
    y_predSVM=clf1.predict(X_test)
    results=cross_val_score(clf1, X_test, y_test, cv=10)
    #print('SVM',accuracy_score(y_test, y_predSVM))
    SVM=clf1.predict(feature)
    m = confusion_matrix(y_test, y_predSVM)
    #print(classification_report(y_test, y_predSVM))
    #print(SVM)
    
    print("Training K Nearest Neighbour...") 
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train) 
    results=cross_val_score(neigh, X_test, y_test, cv=10)
    y_predKNN=neigh.predict(X_test)
    #print('KNN',accuracy_score(y_test, y_predKNN))
    KNN=neigh.predict(feature)
    m = confusion_matrix(y_test, y_predKNN)
    #print(classification_report(y_test, y_predKNN))
    #print(KNN)

    print("Training Naive Bayes...") 
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    results=cross_val_score(clf, X_test, y_test, cv=10)
    y_predNB=clf.predict(X_test)
    NB=clf.predict(feature)
    #print('DT',accuracy_score(y_test, y_predNB))
    m = confusion_matrix(y_test, y_predNB)
    #print(classification_report(y_test, y_predNB))
    #print(NB)

    print('Training Random Forrest...')
    rf = RandomForestClassifier(random_state=0, min_samples_leaf=5)
    rf.fit(X_train, y_train)
    results=cross_val_score(rf, X_test, y_test, cv=10)
    y_predRF=rf.predict(X_test)
    #print('RF',accuracy_score(y_test, y_predRF))
    RF=rf.predict(feature)
    m = confusion_matrix(y_test, y_predRF)
    #print(classification_report(y_test, y_predRF))
    #print(RF)

    
         
    #prediction using consensus
    print("Waiting for results...")
    vote_result=[]
    c_ann=[[0 for x in range(classnum)] for y in range(1)]
    c_svm=[[0 for x in range(classnum)] for y in range(1)]
    c_knn=[[0 for x in range(classnum)] for y in range(1)]
    c_nb=[[0 for x in range(classnum)] for y in range(1)]
    c_rf=[[0 for x in range(classnum)] for y in range(1)]
    
    vote_result = [[0 for x in range(classnum)] for y in range(len(SVM))]
    for i in range(len(RF)):
        for j in range(classnum):
          if round(ANN[i][j])==1.0:
              #print('entered',i)
              vote_result[i][j]=vote_result[i][j]+1
              c_ann[0][j]=c_ann[0][j]+1
                      
          if SVM[i]==j+1:
              vote_result[i][j]=vote_result[i][j]+1
              c_svm[0][j]=c_svm[0][j]+1
          
          if KNN[i]==j+1:
              vote_result[i][j]=vote_result[i][j]+1
              c_knn[0][j]=c_knn[0][j]+1
          
          if NB[i]==j+1:
              vote_result[i][j]=vote_result[i][j]+1
              c_nb[0][j]=c_nb[0][j]+1
          
          if RF[i]==j+1:
              vote_result[i][j]=vote_result[i][j]+1
              c_rf[0][j]=c_rf[0][j]+1
              
##    print(vote_result)
##    print('-------------')
##    print(c_ann)
##    print('-------------')
##    print(c_svm)
##    print('-------------')
##    print(c_knn)
##    print('-------------')
##    print(c_nb)
##    print('-------------')
##    print(c_rf)

    maximum=0
    for i in range(len(vote_result)):
         max=0
         if vote_result[i][0]>=maximum:
                   maximum=vote_result[i][0]
                   win=1
         if vote_result[i][1]>=maximum:
                   maximum=vote_result[i][1]
                   win=2
         if vote_result[i][2]>=maximum:
                   maximum=vote_result[i][2]
                   win=3          
         if vote_result[i][3]>=maximum:
                   maximum=vote_result[i][3]
                   win=4
         if vote_result[i][4]>=maximum:
                   maximum=vote_result[i][4]
                   win=5          
         if win==1:
             print('PDB structure',i+1,' is a Type 3 effector proteins')
         if win==2:
             print('PDB structure',i+1,' is a Type 4 effector proteins')
         if win==3:
             print('PDB structure',i+1,' is a Type 6 effector proteins')
         if win==4:
             print('PDB structure',i+1,' is a Type 1/2/7 effector proteins')
         if win==5:
             print('PDB structure',i+1,' is a non-effector proteins')    
    
def pdbextract(testing):
    import Bio
    from Bio.PDB import PDBParser
    import numpy as np
    import scipy
    from scipy.spatial import Delaunay
    from scipy.spatial import ConvexHull
    import copy
    import sys
    import warnings
    import csv

    if not sys.warnoptions:
      warnings.simplefilter("ignore")

    temp1=[[0 for j in range(19)] for i in range(1)]
    feature=[]
    for i in range(1):
     #filename="H:/rishika/3_work_complex_network_effector/type6/PDB_files/pdb_matlab/t"+str(i+1)+".pdb"
     p = PDBParser()
     s = p.get_structure("non1", testing)
     coordinate=[]
     name=[]
         
     count=0
     for chains in s:
      for chain in chains:
        for residue in chain:                             
            for atom in residue:
                p=atom.get_vector()
                name.append(atom.get_name())
                coordinate.append([p[0], p[1], p[2]])
                count=count+1
     

     #print(len(coordinate))
     coordinate1=coordinate.copy()
     count=convex_hull_layercount(coordinate)
     radius=radius_of_gyration(coordinate)
     pd=packing_density(coordinate, name, radius)
     com=compactness(coordinate1,radius)
     per=surfacecomposition(coordinate1, name)
 
     
     feature.append([count, radius, pd, com, per[0], per[1], per[2], per[3]])
     
    
    with open("3dfeature_non.csv", 'w') as myfile:
       wr = csv.writer(myfile)
       wr.writerows(feature)
    return feature
    
    
def convex_hull_layercount(coordinate):
     import Bio
     from Bio.PDB import PDBParser
     import numpy as np
     import scipy
     from scipy.spatial import Delaunay
     from scipy.spatial import ConvexHull
     import copy
     #convex hull count
     points=np.asarray(coordinate)
     layer_convex_hull=0
     while len(coordinate)>3:
         layer_convex_hull=layer_convex_hull+1
         hull=ConvexHull(points)
         x=[]
         x=np.unique(hull.simplices)
         coordinate1=[]
         coordinate1=coordinate.copy()
         for i in range(len(x)):
             #print(len(coordinate),len(coordinate1))
             temp=coordinate[x[i]]
             #print(x[i])
             coordinate1.remove(temp)
         coordinate=[]      
         coordinate=coordinate1.copy()
         points=np.asarray(coordinate)
     #print("Layers of convex hull: ",layer_convex_hull)
     return layer_convex_hull

def radius_of_gyration(coordinate):
    x=0;y=0;z=0
    
    for i in range(len(coordinate)):
        x=x+float(coordinate[i][0])
        y=y+float(coordinate[i][1])
        z=z+float(coordinate[i][2])
    x=x/len(coordinate); y=y/len(coordinate); z=z/len(coordinate)   
    #print("Mean: ",x/len(coordinate),y/len(coordinate),z/len(coordinate))
    r=0
    for i in range(len(coordinate)):
        r=r+(((x-float(coordinate[i][0]))**2)+((y-float(coordinate[i][1]))**2)+((z-float(coordinate[i][2]))**2))**0.5
    r=r/len(coordinate)
    return r
    #print("Radius of gyration: ", r)

def packing_density(coordinate, name, radius):
    import math
    atom_vol=0
    for i in range(len(name)):
        if name[i][0]=='C':
            atom_vol=atom_vol+(4/3)*math.pi*(0.7**3)
        elif name[i][0]=='O':
            atom_vol=atom_vol+(4/3)*math.pi*(0.6**3)
        elif name[i][0]=='N':
            atom_vol=atom_vol+(4/3)*math.pi*(0.65**3)
        elif name[i][0]=='S':
            atom_vol=atom_vol+(4/3)*math.pi*(1**3)
        elif name[i][0]=='H':
            atom_vol=atom_vol+(4/3)*math.pi*(0.53**3)     
        else:
            atom_vol=atom_vol+(4/3)*math.pi*(0.7**3)   
    protein_vol=(4/3)*math.pi*(radius**3)        
    packingdensity=atom_vol/protein_vol
    #print(atom_vol, protein_vol,packingdensity)
    return packingdensity

def compactness(coordinate,radius):
    from scipy.spatial import ConvexHull
    import numpy as np
    import math
    points=[];hull=[];hull1=[]
    points=np.asarray(coordinate)
    hull1=ConvexHull(points)
    hull=hull1.simplices
    #print(len(hull))
    area=0; sum_area=0
    for i in range(len(hull)):
        x1=coordinate[hull[i][0]][0]; y1=coordinate[hull[i][0]][1]; z1=coordinate[hull[i][0]][2];
        x2=coordinate[hull[i][1]][0]; y2=coordinate[hull[i][1]][1]; z2=coordinate[hull[i][1]][2];
        x3=coordinate[hull[i][2]][0]; y3=coordinate[hull[i][2]][1]; z3=coordinate[hull[i][2]][2];

        t1=x2-x1;t2=y2-y1;t3=z2-z1
        t4=x3-x1;t5=y3-y1;t6=z3-z1

        f1=t2*t6-t3*t5; f2=t1*t6-t3*t4; f3=t1*t5-t2*t4
        area=(((f1**2)+(f2**2)+(f3**2))**0.5)/2
        sum_area=sum_area+area
    #print(sum_area)
    sphere_surface_area=4*math.pi*(radius**2)
    #print(sphere_surface_area)
    return sum_area/sphere_surface_area

def surfacecomposition(coordinate, name):
    from scipy.spatial import ConvexHull
    import numpy as np
    import math
    
    points=[];hull=[];hull1=[]
    points=np.asarray(coordinate)
    hull1=ConvexHull(points)
    hull=hull1.simplices
    p=np.unique(hull)
    countC=0; countO=0; countN=0; countS=0;
    for i in range(len(p)):
        if name[p[i]][0]=='C':
            countC=countC+1
        elif name[p[i]][0]=='O':
            countO=countO+1
        elif name[p[i]][0]=='N':
            countN=countN+1    
        elif name[p[i]][0]=='S':
            countS=countS+1
        else:
            countC=countC+1
    f=[countC/len(p), countO/len(p), countN/len(p), countS/len(p)]    
    return f
