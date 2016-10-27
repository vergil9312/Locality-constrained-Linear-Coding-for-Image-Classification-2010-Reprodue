import scipy.io
import numpy.matlib
import numpy as np
import numpy.linalg
import math
import os.path
import pickle
import sklearn
from sklearn import svm
from LLC_coding_appr import LLC_coding_appr
from LLC_pooling import LLC_pooling


# parameter setting
pyramid = [1, 2, 4]                # spatial block structure for the SPM
knn = 5                            # number of neighbors for local coding
c = 10                             # regularization parameter for linear SVM

nRounds = 10                      # number of random test on the dataset
tr_num  = 30                      # training examples per category
mem_block = 3000                  # maxmum number of testing features loaded each time

# Set path
img_dir = 'image/Caltech101'          
data_dir = 'data/Caltech101'       
fea_dir = 'features/Caltech101'    

# codebook loading
database = scipy.io.loadmat('database.mat')
database = database['database']
imnum = database['imnum'][0][0][0][0]
database_path = ['']*imnum
database_label = database['label'][0][0].astype(int)
for i in range(imnum):
    database_path[i] = database['path'][0][0][0][i][0]

B = scipy.io.loadmat('b.mat')
B = B['B']

nCodebook = B.shape[1]

# extract image features
dFea = sum(np.square(pyramid)*nCodebook)
nFea = len(database_path)
fdatabase_path = ['']*nFea
fdatabase_label = np.zeros((nFea,1))

for iter1 in range(nFea):
    # loading image information
    feaSet = scipy.io.loadmat(database_path[iter1])
    feaSet = feaSet['feaSet']
    X = feaSet['feaArr'][0][0]
    X1 = feaSet['x'][0][0]
    Y = feaSet['y'][0][0]
    img_width = feaSet['width'][0][0][0][0]
    img_height = feaSet['height'][0][0][0][0]
    
    # initializeing information
    fpath = database_path[iter1]
    flabel = database_label[iter1][0]
    [rtpath,fname] = os.path.split(database_path[iter1])
    if not os.path.exists(fea_dir+'/'+flabel.astype(str)):
        os.makedirs(fea_dir+'/'+flabel.astype(str))

    # LLC pooling
    fea = LLC_pooling(B,X,pyramid,knn,img_width,img_height,X1,Y)
    
    #storing information
    label = database_label[iter1][0]
    with open(fea_dir+'/'+str(flabel)+'/'+fname+'.pickle', 'w') as f:
        pickle.dump([fea, label], f)
    fdatabase_label[iter1] = flabel
    fdatabase_path[iter1] = fea_dir+'/'+str(flabel)+'/'+fname+'.pickle'
    


# evualate the peroformance of the image feature using linear SVM 

with open('labelandpath.pickle') as f:
    fdatabase_label, fdatabase_path = pickle.load(f)

clabel = np.unique(fdatabase_label)
nclass = len(clabel)
accuracy = np.zeros((nRounds,1))

for ii in range(nRounds): #repalce 1 with nRounds
    print 'Round:' + str(ii)
    tr_idx = []
    ts_idx = []
    
    for jj in range(nclass):
        idx_label = np.where(fdatabase_label == clabel[jj])[0]
        num = len(idx_label)
        
        idx_rand = np.random.permutation(num)
        
        tr_idx = np.append(tr_idx,idx_label[idx_rand[0:tr_num]])
        ts_idx = np.append(ts_idx,idx_label[idx_rand[tr_num:]])
    
    print 'Training number : ' + str(len(tr_idx))
    print 'Testing number: ' + str(len(ts_idx))
    
    # load the training features
    tr_fea = np.zeros((len(tr_idx),dFea))
    tr_label = np.zeros((len(tr_idx),1))
    
    for jj in range(len(tr_idx)):
        fpath = fdatabase_path[int(tr_idx[jj])]
        with open(fpath) as f:
            fea, label = pickle.load(f)
        tr_fea[jj,:] = np.transpose(fea)
        tr_label[jj] = label
    
    lin_clf = svm.LinearSVC()
    lin_clf.fit(tr_fea,tr_label)
    del tr_fea
    
    # load the testing features
    ts_num = len(ts_idx)
    ts_label = []
    
    # load the testing features directly into memory for testing 
    ts_fea = np.zeros((len(ts_idx),dFea))
    ts_label = np.zeros((len(ts_idx),1))
    
    for jj in range(len(ts_idx)):
        fpath = fdatabase_path[int(ts_idx[jj])]
        with open(fpath) as f:
            fea, label = pickle.load(f)
        ts_fea[jj,:] = np.transpose(fea)
        ts_label[jj] = label
    
    C = lin_clf.predict(ts_fea)
    
    # normalize the classification accuracy by averaging over different classes
    acc = np.zeros((nclass,1))
    
    for jj in range(nclass):
        c = clabel[jj]
        idx = np.where(ts_label == c)[0]
        curr_pred_label = np.transpose(np.matrix(C[idx]))
        curr_gnd_label = ts_label[idx]
        acc[jj] = float(len(np.where(curr_pred_label == curr_gnd_label)[0])) / float(len(idx))
    
    accuracy[ii] = np.mean(acc)

Ravg = np.mean(accuracy)
Rstd = np.std(accuracy)
        
    
        
