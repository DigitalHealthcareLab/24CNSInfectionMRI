#%%
import os
import torch
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import numpy as np

from config import *
from labels import *
from Inference.inference_lookup import inference_lookup_cnn
from src.model_densenet_vectoroutput import d169_3d
from src.seed import seed_everything
from src.testing_vectoroutput import test_epoch_vectoroutput
from src.utils import Logger

###############################################################################
# tmux[n] = inner[n + 1] / labels[n + 1]
labels = inner4
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
pathogenlist = [0]
y_ext_dir = '/home/yhchoi/24BrainMRI_Data/Labels/External/prognosis_labels_external.pt'
tensors_ext_dir = '/home/yhchoi/24BrainMRI_Data/Tensors/External/'
folds_ext_dir = '/home/yhchoi/24BrainMRI_Data/Labels/External/pathogen_indices_'
filenames_ext_dir = '/home/yhchoi/24BrainMRI_Data/Labels/External/filename_labels_external.pt'
###############################################################################
#TODO Tidy up directories
seed_everything(random_state)
y_nlr = torch.load(y_ext_dir)
y_nlr = torch.tensor([1 if item in labels_to_classify else 0 for item in y_nlr])

for pathogen in pathogenlist:
    for part_to_segment in labels:
        labels_nlr = label_lookups(part_to_segment, 'left') == label_lookups(part_to_segment, 'right')
        part_to_segment = part_to_segment.replace('-', '_')

        # load the data
        tensordir = f'{tensors_ext_dir}{part_to_segment.upper()}_TENSORS/'
        X = torch.load(tensordir + part_to_segment.lower() + '_external.pt')

        # concatenate the data
        if labels_nlr:
            y = y_nlr
            fold_indices = [torch.load(folds_ext_dir + str(i) + '_external.pt') for i in pathogenlist]
            filenames = torch.load(filenames_ext_dir)
        else:
            y = torch.cat([y_nlr, y_nlr], dim=0)
            fold_indices = [torch.cat([torch.load(folds_ext_dir + str(i) + '_external.pt'),
                            torch.add(torch.load(folds_ext_dir + str(i) + '_external.pt'), len(y_nlr))]) for i in pathogenlist]
            filenames = torch.load(filenames_ext_dir) + \
                        torch.load(filenames_ext_dir)

        model = d169_3d(num_classes=number_of_classes, sample_size=64, sample_duration=96, norm="bn", act="lrelu")
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        bestmodel_path = Path(f'/home/yhchoi/FSTrain_0205/Inference/cnn_all/{pathogen}/{part_to_segment}_bestmodel.pt')
        model.load_state_dict(torch.load(bestmodel_path))

        idx_test = fold_indices[pathogen]

        if labels_nlr:
            filenames_test = [filenames[i] for i in idx_test]
            filenames_test_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors/{pathogen}/filenames_test.pt'
            os.makedirs(os.path.dirname(filenames_test_save_path), exist_ok=True)
            torch.save(filenames_test, filenames_test_save_path)

            labels_test = [y[i] for i in idx_test]
            labels_test_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors/{pathogen}/labels_test.pt'
            os.makedirs(os.path.dirname(labels_test_save_path), exist_ok=True)
            torch.save(labels_test, labels_test_save_path)

            X_test = X[fold_indices[pathogen]]
            y_test = y[fold_indices[pathogen]]

            X_test = X_test.to(device)

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            ######################################################################
            # TEST SET
            ######################################################################
            feat_vecs = test_epoch_vectoroutput(test_loader, model, criterion, device)

            feat_vecs_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors/{pathogen}/{part_to_segment}_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_save_path), exist_ok=True)

            torch.save(feat_vecs, feat_vecs_save_path)

        else:
            X_test = X[fold_indices[pathogen]]
            y_test = y[fold_indices[pathogen]]

            X_test_l = X_test[:int(X_test.shape[0]/2), :, :, :, :]
            y_test_l = y_test[:int(X_test.shape[0]/2)]
            
            X_test_l = X_test_l.to(device)

            ######################################################################
            # LEFT TEST SET
            ######################################################################
            test_dataset_l = TensorDataset(X_test_l, y_test_l)
            test_loader_l = DataLoader(test_dataset_l, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_l = test_epoch_vectoroutput(test_loader_l, model, criterion, device)

            feat_vecs_l = torch.tensor(np.array(feat_vecs_l))

            feat_vecs_l_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors/{pathogen}/{part_to_segment}_left_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_l_save_path), exist_ok=True)

            torch.save(feat_vecs_l, feat_vecs_l_save_path)

            ######################################################################
            # RIGHT TEST SET
            ######################################################################
            X_test_r = X_test[int(X_test.shape[0]/2):, :, :, :, :]
            y_test_r = y_test[int(X_test.shape[0]/2):]
            
            X_test_r = X_test_r.to(device)

            test_dataset_r = TensorDataset(X_test_r, y_test_r)
            test_loader_r = DataLoader(test_dataset_r, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_r = test_epoch_vectoroutput(test_loader_r, model, criterion, device)

            feat_vecs_r = torch.tensor(np.array(feat_vecs_r))

            feat_vecs_r_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors/{pathogen}/{part_to_segment}_right_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_r_save_path), exist_ok=True)

            torch.save(feat_vecs_r, feat_vecs_r_save_path)

print('Done')
