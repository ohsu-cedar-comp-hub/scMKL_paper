import numpy as np
from scipy.sparse import load_npz
import argparse
import os
import pickle
import time
import tracemalloc
import sys

import scmkl # This will be run with scMKL conda version now

parser = argparse.ArgumentParser(description='Unimodal classification of single cell data with hallmark prior information')
parser.add_argument('-d', '--dataset', help = 'Which dataset to classify', choices = ['prostate',  'MCF7', 'T47D', 'lymphoma', 'song_prostate'], type = str)
parser.add_argument('-a', '--assay', help = 'Which assay to use', type = str, choices = ['rna', 'atac', 'gene_scores'])
parser.add_argument('-r', '--replication', help = 'Which replication to use', type = int, default = 1)
parser.add_argument('-l', '--lsi', help = 'Should LSI be performed per pathway? *NOTE: can only be used with ATAC*', type = bool, default = False)
parser.add_argument('-n', '--n_cells', help = 'Number of random features to include in the classification', default= 6438, type = int)

args = parser.parse_args()

dataset = args.dataset
assay = args.assay
replication = args.replication
lsi_bool = args.lsi
n_cells = args.n_cells


if lsi_bool:
    assert assay.lower() == 'atac', 'TFIDF can only be used with ATAC'

output_assay = assay if not lsi_bool else f'{assay}_lsi'

########################################################################

if lsi_bool: 
    reduction = 'SVD'
    tfidf_bool = True

    output_dir = f'/home/groups/CEDAR/scMKL/results/scalability_tests/atac/scMKL/LSI/{n_cells}'

else:
    reduction = None
    tfidf_bool = False
    output_dir = f'/home/groups/CEDAR/scMKL/results/scalability_tests/{assay}/scMKL/{n_cells}'

########################################################################
filename = f'Replication_{replication}.pkl'


########################################################################
## Change this for multimodal experiments
X = load_npz(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_scipy.npz')
feature_names = np.load(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_feature_names.npy', allow_pickle= True)
########################################################################

cell_labels = np.load(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_cell_metadata.npy', allow_pickle= True)

random_cells = np.random.default_rng(replication * 100).choice(np.arange(len(cell_labels)), n_cells, replace = False)

X = X[random_cells,:]
cell_labels = cell_labels[random_cells]

########################################################################
## Change this for desired group_dict
with open(f'/home/groups/CEDAR/scMKL/data/{dataset}/{dataset}_{assay.upper()}_hallmark_groupings.pkl', 'rb') as fin:
    group_dict = pickle.load(fin)
########################################################################

if assay == 'atac' and not lsi_bool:
    kernel_type = 'Laplacian'
    scale_data = False
    distance_metric = 'cityblock'
else:
    kernel_type = 'Gaussian'
    scale_data = True
    distance_metric = 'euclidean'

########################################################################
## Alpha sets may need to change depending on dataset/experiment/groupings/assay

if assay == 'rna':
    alpha_list = np.round(np.linspace(3.1, 0.1, 10), 2)
elif assay == 'atac':
    alpha_list = np.round(np.linspace(1.9, 0.1, 10), 2)
########################################################################

########################################################################
## May need to change for number of groupings for scalability
D = int(np.sqrt(len(cell_labels)) * np.log(np.log(len(cell_labels))))
########################################################################

adata = scmkl.create_adata(X = X, feature_names = feature_names, cell_labels = cell_labels, group_dict = group_dict,
                         scale_data = scale_data, D = D, remove_features = True, kernel_type= kernel_type, 
                         distance_metric= distance_metric, class_threshold = None, random_state = 100 * replication,
                         reduction = reduction, tfidf = tfidf_bool)


tracemalloc.start()
sigma_start = time.time()


print('Estimating Sigma', flush = True)
adata = scmkl.estimate_sigma(adata, n_features = 200 if not lsi_bool else 10000)

print('Calculating Z', flush = True)
adata = scmkl.calculate_z(adata, n_features = 5000 if not lsi_bool else 10000)


z_end = time.time()

print(z_end - sigma_start)

# _, min_alpha = scmkl.optimize_sparsity(adata, target = 50, group_size = 2 * D, starting_alpha = 0.1, increment = 0.05)
# max_alpha = min_alpha + 0.8

# alpha_list = np.round(np.linspace(max_alpha, min_alpha, 19), 2)

# alpha_star = scmkl.optimize_alpha(adata, group_size= 2 * D, alpha_array= alpha_list, k = 4)
alpha_star = 0.1

group_names = list(group_dict.keys())
train_start = time.time()

adata = scmkl.train_model(adata, group_size= 2*D, alpha = alpha_star)

train_end = time.time()

predicted, metrics = scmkl.predict(adata, metrics = ['AUROC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'])
selected_pathways = scmkl.find_selected_groups(adata)
group_norms = [np.linalg.norm(adata.uns['model'].coef_[i * 2 * D: (i + 1) * 2 * D - 1]) for i in np.arange(len(group_names))]

results = {}
results['Metrics'] = metrics
results['Selected_pathways'] = selected_pathways
results['Norms'] = group_norms
results['Predictions'] = predicted
results['Observed'] = adata.obs['labels'].iloc[adata.uns['test_indices']]
results['Test_indices'] = adata.uns['test_indices']
results['Group_names']= group_names
results['Model'] = adata.uns['model']
results['Inference_time'] = z_end - sigma_start + (train_end - train_start) # Time to calculate sigma and Z's plus mean train/test test time all alphas
results['RAM_usage'] = f'{tracemalloc.get_traced_memory()[1] / 1e9} GB'
results['Alpha_star'] = alpha_star

print(f'Memory Usage: {tracemalloc.get_traced_memory()[1] / 1e9} GB')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# with open(f'{output_dir}/{filename}', 'wb') as fout:
#     pickle.dump(results, fout)
