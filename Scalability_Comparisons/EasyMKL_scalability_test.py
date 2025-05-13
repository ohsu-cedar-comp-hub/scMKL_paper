import numpy as np
from MKLpy.algorithms import EasyMKL
from scipy.sparse import load_npz
from scipy.spatial.distance import cdist
import torch
import argparse
import os
import time
import tracemalloc
import sys
import pickle
import scipy
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

tracemalloc.start()

parser = argparse.ArgumentParser(description='Run classification for RNA/ATAC data with EasyMKL from MKLpy package with a kernel per hallmark pathway.')
parser.add_argument('-r' ,'--replication', help = 'Which replication to run', type = int, default = 1)
parser.add_argument('-d', '--dataset', help = 'Which dataset to use', type = str, default = 'MCF7')
parser.add_argument('-a', '--assay', help = 'Which assay to use', type = str, choices = ['rna', 'atac'])
parser.add_argument('-n', '--n_cells', help = 'Number of random features to include in the classification', default = 6438, type = int)

args = parser.parse_args()

dataset = args.dataset
replication = args.replication
n_cells = args.n_cells
assay = args.assay

X = load_npz(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_scipy.npz')
group_dict = np.load(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_hallmark_groupings.pkl', allow_pickle = True)
feature_names = np.load(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_feature_names.npy', allow_pickle = True)
labels = np.load(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_cell_metadata.npy', allow_pickle = True)

output_dir = f'/home/groups/CEDAR/scMKL/results/scalability_tests/{assay}/EasyMKL/{n_cells}'
filename = f'Replication_{replication}.pkl'

metric = 'euclidean' if assay == 'rna' else 'cityblock'


def Filter_Features(X, feature_set, group_dict):
    '''
    Function to remove unused features from X matrix.  Any features not included in group_dict will be removed from the matrix.
    Also puts the features in the same relative order (of included features)
    Input:
            X- Data array. Can be Numpy array or Scipy Sparse Array
            feature_names- Numpy array of corresponding feature names
            group_dict- Dictionary containing feature grouping information.
                        Example: {geneset: np.array(gene_1, gene_2, ..., gene_n)}
    Output:
            X- Data array containing data only for features in the group_dict
            feature_names- Numpy array of corresponding feature names from group_dict
    '''
    assert X.shape[1] == len(feature_set), 'Given features do not correspond with features in X'    

    group_features = set()

    # Store all objects in dictionary in array
    for group in group_dict.keys():
        group_features.update(set(group_dict[group]))

    # Find location of desired features in whole feature set
    group_feature_indices = np.nonzero(np.isin(feature_set, np.array(list(group_features)), assume_unique = True))[0]

    # Subset only the desired features and their data
    X = X[:,group_feature_indices]
    feature_set = feature_set[group_feature_indices]

    return X, feature_set

def Sparse_Var(X, axis = None):

    '''
    Function to calculate variance on a sparse matrix.
    Input:
        X- A scipy sparse or numpy array
        axis- Determines which axis variance is calculated on. Same usage as Numpy
            axis = 0 => column variances
            axis = 1 => row variances
            axis = None => total variance (calculated on all data)
    Output:
        var- Variance values calculated over the given axis
    '''

    # E[X^2] - E[X]^2
    if scipy.sparse.issparse(X):
        var = np.array((X.power(2).mean(axis = axis)) - np.square(X.mean(axis = axis)))
    else:
        var = np.var(X, axis = axis)
    return var.ravel()

def process_data(X_train, X_test):

    '''
    Function to preprocess data matrix according to type of data (counts- e.g. rna, or binary- atac)
    Will process test data according to parameters calculated from test data

    Input:
        X_train- A scipy sparse or numpy array
        X_train- A scipy sparse or numpy array
        data_type- 'counts' or 'binary'.  Determines what preprocessing is applied to the data. 
            Log transforms and standard scales counts data
            TFIDF filters ATAC data to remove uninformative columns
    Output:
        X_train, X_test- Numpy arrays with the process train/test data respectively.
    '''

    # Remove features that have no variance in the training data (will be uniformative)

    var = np.var(X_train, axis = 0)
    variable_features = np.where(var > 1e-5)[0]

    X_train = X_train[:,variable_features]
    X_test = X_test[:, variable_features]

    if scipy.sparse.issparse(X_train):
        X_train = X_train.log1p()
        X_test = X_test.log1p()
    else:
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)
        
    #Center and scale count data
    train_means = np.mean(X_train, 0)
    train_sds = np.sqrt(var[variable_features])

    X_train = (X_train - train_means) / train_sds
    X_test = (X_test - train_means) / train_sds

    return X_train, X_test

def Train_Test_Split(y, seed_obj = np.random.default_rng(100), train_ratio = 0.8):
    '''
    Function to calculate training and testing indices for given dataset, preserving the ratio of each class in y.
    Input:
            y- Numpy array of cell labels. Can have any number of classes for this function.
            seed_obj- Numpy random state used for random processes. Can be specified for reproducubility or set by default.
            train_ratio- decimal value ratio of features in training:testing sets
    Output:
            train_indices- Array of indices of training cells
            test_indices- Array of indices of testing cells
    '''

    unique_labels = np.unique(y)
    train_indices = []

    for label in unique_labels:

        # Find index of each unique label
        label_indices = np.where(y == label)[0]

        # Sample these indices according to train ratio
        train_label_indices = seed_obj.choice(label_indices, int(len(label_indices) * train_ratio), replace = False)
        train_indices.extend(train_label_indices)

    # Test indices are the indices not in the train_indices
    test_indices = np.setdiff1d(np.arange(len(y)), train_indices, assume_unique = True)

    return train_indices, test_indices

def TF_IDF_filter(X, mode = 'filter'):
    '''
    Function to use Term Frequency Inverse Document Frequency filtering for atac data to find meaningful features. 
    If input is pandas data frame or scipy sparse array, it will be converted to a numpy array.
    Input:
            x- Data matrix of cell x feature.  Must be a Numpy array or Scipy sparse array.
            mode- Argument to determine what to return.  Must be filter or normalize
    Output:
            TFIDF- Output depends on given 'mode' parameter
                'filter'- returns which column sums are non 0 i.e. which features are significant
                'normalize'- returns TFIDF filtered data matrix of the same dimensions as x. Returns as scipy sparse matrix
    '''

    assert mode in ['filter', 'normalize'], 'mode must be "filter" or "normalize".'
    
    if scipy.sparse.issparse(X):
        row_sum = np.array(X.sum(axis=1)).flatten()
        tf = scipy.sparse.csc_array(X / row_sum[:, np.newaxis])
        doc_freq = np.array(np.sum(X > 0, axis=0)).flatten()
    else:
        row_sum = np.sum(X, axis=1, keepdims=True)
        tf = X / row_sum    
        doc_freq = np.sum(X > 0, axis=0)

    idf = np.log1p((1 + X.shape[0]) / (1 + doc_freq))
    tfidf = tf * idf

    if mode == 'normalize':
        if scipy.sparse.issparse(tfidf):
            tfidf = scipy.sparse.csc_matrix(tfidf)
        return tfidf
    elif mode == 'filter':
        significant_features = np.where(np.sum(tfidf, axis=0) > 0)[0]
        return significant_features

def gaussian_kernels(X_train, X_test, metric = 'euclidean'):
    D_train = cdist(X_train, X_train, metric = metric)
    D_test = cdist(X_test, X_train, metric = metric)

    sigma_train = np.mean(D_train)

    K_train = np.exp( -(D_train)**2 / (2 * sigma_train**2))
    K_test = np.exp( -(D_test)**2 / (2 * sigma_train**2))

    return K_train, K_test

X, feature_names = Filter_Features(X, feature_names, group_dict)

X = np.array(X.todense().astype(np.float32))

random_cells = np.random.default_rng(replication * 100).choice(np.arange(len(labels)), n_cells, replace = False)

X = X[random_cells,:]
labels = labels[random_cells]

y = np.zeros(labels.shape).astype(np.float32)


for i, label in enumerate(np.unique(labels)):
    y[labels == label] = i

train_indices, test_indices = Train_Test_Split(y, seed_obj= np.random.default_rng(100 * replication))

X_train = X[train_indices, :]
X_test = X[test_indices, :]

y_train = y[train_indices]
y_test = y[test_indices]

lambda_values = np.linspace(0, 1, 5)
k = 4

# Cross Validation
positive_indices = np.where(y_train == np.unique(y_train)[0])[0]
negative_indices = np.setdiff1d(np.arange(len(y_train)), positive_indices)

positive_annotations = np.arange(len(positive_indices)) % k
negative_annotations = np.arange(len(negative_indices)) % k

auroc_array = np.zeros((k, len(lambda_values)))

for fold in np.arange(k):
    fold_train = np.concatenate((positive_indices[np.where(positive_annotations != fold)[0]], negative_indices[np.where(negative_annotations != fold)[0]]))
    fold_test = np.concatenate((positive_indices[np.where(positive_annotations == fold)[0]], negative_indices[np.where(negative_annotations == fold)[0]]))    

    KLtr = []
    KLte = []
    y_train_fold = y_train[fold_train]
    y_test_fold = y_train[fold_test]



    for i, features in enumerate(group_dict.values()):
        if i % 10 == 0:
            print('*', flush= True, end='')

        pathway_X = X_train[:, np.where(np.isin(feature_names, np.array(list(features))))[0]]

        fold_train_X = pathway_X[fold_train, :]
        fold_test_X = pathway_X[fold_test, :]

        fold_train_X, fold_test_X = process_data(fold_train_X, fold_test_X)

        fold_train_K, fold_test_K = gaussian_kernels(fold_train_X, fold_test_X, metric)

        KLtr.append(torch.DoubleTensor(fold_train_K))
        KLte.append(torch.DoubleTensor(fold_test_K))

    for n in np.arange(len(lambda_values)):

        model = EasyMKL(lam= lambda_values[n]).fit(KLtr, y_train_fold)


        pred = model.predict(KLte)
        scores = model.decision_function(KLte)

        end = time.time()

        auroc = roc_auc_score(y_test_fold, scores)

        auroc_array[fold, n] = auroc

opt_lambda = lambda_values[np.argmax(np.mean(auroc_array, axis = 0))]

start = time.time()

KLtr = []
KLte = []

for i, features in enumerate(group_dict.values()):
    if i % 10 == 0:
        print('*', flush= True, end='')

    pathway_X = X[:, np.where(np.isin(feature_names, np.array(list(features))))[0]]

    X_train = pathway_X[train_indices, :]
    X_test = pathway_X[test_indices, :]

    X_train, X_test = process_data(X_train, X_test)

    K_train, K_test = gaussian_kernels(X_train, X_test, metric)

    KLtr.append(torch.DoubleTensor(K_train))
    KLte.append(torch.DoubleTensor(K_test))

# Here we use custom kernel functions because we believe that MKLpy kernel functions lead to data leakage


model = EasyMKL(lam = opt_lambda).fit(KLtr, y_train)


pred = model.predict(KLte)
scores = model.decision_function(KLte)

end = time.time()


accuracy = accuracy_score(y_test, pred)
auroc = roc_auc_score(y_test, scores)
precision = precision_score(y_test, pred, pos_label= np.unique(y_test)[0])
recall = recall_score(y_test, pred, pos_label= np.unique(y_test)[0])
f1 = f1_score(y_test, pred, pos_label= np.unique(y_test)[0])

metrics = {'AUROC': auroc,
           'F1-Score': f1, 
           'Accuracy': accuracy, 
           'Precision': precision, 
           'Recall': recall}

results = {}
results['Metrics'] = metrics
results['Predictions'] = pred
results['Inference_time'] = end - start
results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'
results['Model_weights'] = {list(group_dict.keys())[i]: float(model.solution.weights[i]) for i in np.arange(len(group_dict))}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(f'{output_dir}/{filename}', 'wb') as fout:
    pickle.dump(results, fout)
