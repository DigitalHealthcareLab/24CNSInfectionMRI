import os
import torch
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import numpy as np

from .src._cnn_config import *
from ._labels import *
from src.model_densenet_vectoroutput import d169_3d
from src.seed import seed_everything
from src.testing_vectoroutput import test_epoch_vectoroutput
from src.utils import Logger

###############################################################################
# tmux[n] = inner[n + 1] / labels[n + 1]
# labels = labels11
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
pathogenlist = [0]
###############################################################################

seed_everything(random_state)
y_nlr = torch.load(y_dir)
y_nlr = torch.tensor([1 if item in labels_to_classify else 0 for item in y_nlr])

for pathogen in pathogenlist:
    for part_to_segment in labels:
        labels_nlr = label_lookups(part_to_segment, 'left') == label_lookups(part_to_segment, 'right')
        part_to_segment = part_to_segment.replace('-', '_')

        # load the data
        tensordir = tensordir_dir + part_to_segment.upper() + '_TENSORS/'
        X = torch.load(tensordir + part_to_segment.lower() + '_internal.pt')

        #TODO : add folddir
        if labels_nlr:
            y = y_nlr
            fold_indices = [torch.load(homedir + '/src/folds/kfoldsplitindices_' + str(random_state) +'_' + str(i) + '.pt') for i in pathogenlist]
        else:
            y = torch.cat([y_nlr, y_nlr], dim=0)
            fold_indices = [torch.load(homedir + '/src/folds/kfoldsplitindices_' + str(random_state) +'_' + str(i) + '.pt') + \
                            [torch.add(sublist, len(y_nlr)) for sublist in torch.load(homedir + '/src/folds/kfoldsplitindices_' + \
                                                                                        str(random_state) +'_' + str(i) + '.pt')]
                            for i in pathogenlist]

        partlogdir = homedir + '/TestLogs/' + part_to_segment.upper() + 'LOGS'       
        pathogenlogdir = partlogdir + '/' + str(pathogen)
        
        model = d169_3d(num_classes=number_of_classes, sample_size=64, sample_duration=96, norm="bn", act="lrelu")
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        bestmodel_path = Path(f'/home/yhchoi/FSTrain_0205/Inference/cnn_all/{pathogen}/{part_to_segment}_bestmodel.pt')
        model.load_state_dict(torch.load(bestmodel_path))
        filenames = torch.load(filenames_dir)

        idx_folds = [fold_indices[pathogen][i] for i in range(num_test_folds)]
        idx_test = idx_folds[test_fold]
        rest_indices = idx_folds[:test_fold] + (idx_folds[test_fold + 1:] if test_fold < len(idx_folds) - 1 else [])
        rest_idx = [item for sublist in rest_indices for item in sublist]

        filenames_test = [filenames[i] for i in idx_test]
        filenames_test_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/filenames_test.pt'
        os.makedirs(os.path.dirname(filenames_test_save_path), exist_ok=True)
        torch.save(filenames_test, filenames_test_save_path)

        filenames_train = [filenames[i] for i in rest_idx]
        filenames_train_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/filenames_train.pt'
        os.makedirs(os.path.dirname(filenames_train_save_path), exist_ok=True)
        torch.save(filenames_train, filenames_train_save_path)

        labels_test = [y[i] for i in idx_test]
        labels_test_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/labels_test.pt'
        os.makedirs(os.path.dirname(labels_test_save_path), exist_ok=True)
        torch.save(labels_test, labels_test_save_path)

        labels_train = [y[i] for i in rest_idx]
        labels_train_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/labels_train.pt'
        os.makedirs(os.path.dirname(labels_train_save_path), exist_ok=True)
        torch.save(labels_train, labels_train_save_path)

        if labels_nlr:
            X_folds = [X[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            y_folds = [y[fold_indices[pathogen][i]] for i in range(num_test_folds)]

            X_test = X_folds[test_fold]
            y_test = y_folds[test_fold]

            X_test = X_test.to(device)
            rest_of_X = X_folds[:test_fold] + (X_folds[test_fold + 1:] if test_fold < len(X_folds) - 1 else [])
            rest_of_y = y_folds[:test_fold] + (y_folds[test_fold + 1:] if test_fold < len(y_folds) - 1 else [])
            rest_indices = idx_folds[:test_fold] + (idx_folds[test_fold + 1:] if test_fold < len(idx_folds) - 1 else [])

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            ######################################################################
            # TEST SET
            ######################################################################
            feat_vecs = test_epoch_vectoroutput(test_loader, model, criterion, device)

            feat_vecs_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_save_path), exist_ok=True)

            torch.save(feat_vecs, feat_vecs_save_path)

            ######################################################################
            # TRAIN SET
            ######################################################################
            train_dataset = TensorDataset(torch.cat(rest_of_X).to(device), torch.cat(rest_of_y))
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_train = test_epoch_vectoroutput(train_loader, model, criterion, device)

            feat_vecs_train_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_featurevectors_train.pt'

            os.makedirs(os.path.dirname(feat_vecs_train_save_path), exist_ok=True)

            torch.save(feat_vecs_train, feat_vecs_train_save_path)

        else:
            X_folds_l = [X[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            X_folds_r = [X[fold_indices[pathogen][i + num_test_folds]] for i in range(num_test_folds)]
            y_folds_l = [y[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            y_folds_r = [y[fold_indices[pathogen][i + num_test_folds]] for i in range(num_test_folds)]

            X_test_l = X_folds_l[test_fold]
            y_test_l = y_folds_l[test_fold]

            X_test_l = X_test_l.to(device)
            rest_of_X_l = X_folds_l[:test_fold] + (X_folds_l[test_fold + 1:] if test_fold < len(X_folds_l) - 1 else [])
            rest_of_y_l = y_folds_l[:test_fold] + (y_folds_l[test_fold + 1:] if test_fold < len(y_folds_l) - 1 else [])

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

            feat_vecs_l_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_left_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_l_save_path), exist_ok=True)

            torch.save(feat_vecs_l, feat_vecs_l_save_path)

            X_test_r = X_folds_r[test_fold]
            y_test_r = y_folds_r[test_fold]
            
            X_test_r = X_test_r.to(device)
            rest_of_X_r = X_folds_r[:test_fold] + (X_folds_r[test_fold + 1:] if test_fold < len(X_folds_r) - 1 else [])
            rest_of_y_r = y_folds_r[:test_fold] + (y_folds_r[test_fold + 1:] if test_fold < len(y_folds_r) - 1 else [])
            rest_indices = idx_folds[:test_fold] + (idx_folds[test_fold + 1:] if test_fold < len(idx_folds) - 1 else [])

            ######################################################################
            # RIGHT TEST SET
            ######################################################################
            test_dataset_r = TensorDataset(X_test_r, y_test_r)
            test_loader_r = DataLoader(test_dataset_r, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_r = test_epoch_vectoroutput(test_loader_r, model, criterion, device)

            feat_vecs_r = torch.tensor(np.array(feat_vecs_r))

            feat_vecs_r_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_right_featurevectors_test.pt'

            os.makedirs(os.path.dirname(feat_vecs_r_save_path), exist_ok=True)

            torch.save(feat_vecs_r, feat_vecs_r_save_path)

            ######################################################################
            # LEFT TRAIN SET
            ######################################################################
            train_dataset_l = TensorDataset(torch.cat(rest_of_X_l).to(device), torch.cat(rest_of_y_l))
            train_loader_l = DataLoader(train_dataset_l, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_train_l = test_epoch_vectoroutput(train_loader_l, model, criterion, device)

            feat_vecs_train_l = torch.tensor(np.array(feat_vecs_train_l))

            feat_vecs_train_l_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_left_featurevectors_train.pt'

            os.makedirs(os.path.dirname(feat_vecs_train_l_save_path), exist_ok=True)

            torch.save(feat_vecs_train_l, feat_vecs_train_l_save_path)

            ######################################################################
            # RIGHT TRAIN SET
            ######################################################################
            train_dataset_r = TensorDataset(torch.cat(rest_of_X_r).to(device), torch.cat(rest_of_y_r))
            train_loader_r = DataLoader(train_dataset_r, batch_size = batch_size, shuffle=False)

            memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
            memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
            print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

            feat_vecs_train_r = test_epoch_vectoroutput(train_loader_r, model, criterion, device)

            feat_vecs_train_r = torch.tensor(np.array(feat_vecs_train_r))

            feat_vecs_train_r_save_path = f'/home/yhchoi/FSTrain_0205/samplevectors_internal/{pathogen}/{part_to_segment}_right_featurevectors_train.pt'

            os.makedirs(os.path.dirname(feat_vecs_train_r_save_path), exist_ok=True)

            torch.save(feat_vecs_train_r, feat_vecs_train_r_save_path)

print('Done')