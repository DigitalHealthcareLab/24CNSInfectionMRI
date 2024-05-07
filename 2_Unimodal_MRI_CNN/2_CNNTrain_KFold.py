import os
import torch
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler

from _labels import *
from src._cnn_config import *
from src.model_densenet import d169_3d
from src.training import train_epoch
from src.validation_withES import val_epoch
from src.seed import seed_everything
from src.testing import test_epoch
from src.earlystopping import EarlyStopping
from src.utils import Logger

###############################################################################
# tmux[n] = inner[n + 1] / labels[n + 1]
# labels = labels1 + labels2 + labels3
# labels = labels4 + labels8 + labels9
# labels = inner0 + inner1 + inner2 + inner3 + inner4 + inner5 + inner6 + inner7 + inner8 + inner9 + labels1 + labels2 + labels3 + labels10
# labels = labels5 + labels6 + labels7 + labels4 + labels8 + labels9 + labels11
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
homedir_alt = homedir
###############################################################################

seed_everything(random_state)
#TODO: y_dir = '/mnt/project/intern3_data/FastSurfer/Extracted/_LABELTENSORS/prognosis_labels.pt'
y_nlr = torch.load(y_dir)
y_nlr = torch.tensor([1 if item in labels_to_classify else 0 for item in y_nlr])

for part_to_segment in labels:
    labels_nlr = label_lookups(part_to_segment, 'left') == label_lookups(part_to_segment, 'right')
    part_to_segment = part_to_segment.replace('-', '_')

    # load the data
    #TODO : tensordir = '/mnt/project/intern3_data/FastSurfer/Extracted/_LABELTENSORS/'
    # tidy up directories based on BADDIE input
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
    
    partlogdir = homedir_alt + '/TrainLogs/' + part_to_segment.upper() + 'LOGS'
    if not os.path.exists(partlogdir):
        os.mkdir(partlogdir)
    for pathogen in pathogenlist:
        pathogenlogdir = partlogdir + '/' + str(pathogen)
        if not os.path.exists(pathogenlogdir):
            os.mkdir(pathogenlogdir)
            
        if labels_nlr:
            X_folds = [X[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            y_folds = [y[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            idx_folds = [fold_indices[pathogen][i] for i in range(num_test_folds)]
        else:
            X_folds_l = [X[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            X_folds_r = [X[fold_indices[pathogen][i + num_test_folds]] for i in range(num_test_folds)]
            y_folds_l = [y[fold_indices[pathogen][i]] for i in range(num_test_folds)]
            y_folds_r = [y[fold_indices[pathogen][i + num_test_folds]] for i in range(num_test_folds)]
            indices_l = [fold_indices[pathogen][i] for i in range(num_test_folds)]
            indices_r = [fold_indices[pathogen][i + num_test_folds] for i in range(num_test_folds)]
            
            X_folds = []
            y_folds = []
            idx_folds = []
            for i in range(num_test_folds):
                X_fold = torch.cat([X_folds_l[i], X_folds_r[i]], dim = 0)
                y_fold = torch.cat([y_folds_l[i], y_folds_r[i]], dim = 0)
                idx_fold = torch.cat([indices_l[i], indices_r[i]], dim = 0)
                X_folds.append(X_fold)
                y_folds.append(y_fold)
                idx_folds.append(idx_fold)

        X_test = X_folds[test_fold]
        y_test = y_folds[test_fold]
        idx_test = idx_folds[test_fold]

        X_test = X_test.to(device)
        rest_of_X = X_folds[:test_fold] + (X_folds[test_fold + 1:] if test_fold < len(X_folds) - 1 else [])
        rest_of_y = y_folds[:test_fold] + (y_folds[test_fold + 1:] if test_fold < len(y_folds) - 1 else [])
        rest_indices = idx_folds[:test_fold] + (idx_folds[test_fold + 1:] if test_fold < len(idx_folds) - 1 else [])

        for val_fold in range(num_val_folds):
                valfolddir = pathogenlogdir + '/val_fold_' + str(val_fold)
                if not os.path.exists(valfolddir):
                    os.mkdir(valfolddir)

                start_index = val_fold * idxnum
                end_index = min((val_fold + 1) * idxnum, len(rest_of_X))
                
                # Validation Data
                X_val_list = rest_of_X[start_index:end_index]
                X_val = torch.cat(X_val_list, dim=0)
                X_val = X_val.to(device)
                y_val_list = rest_of_y[start_index:end_index]
                y_val = torch.cat(y_val_list, dim=0)

                # Validation Indices
                idx_val_list = rest_indices[start_index:end_index]
                idx_val = torch.cat(idx_val_list, dim=0)
                
                # Train Data
                X_train_list = rest_of_X[:start_index] + (rest_of_X[end_index:] if end_index < len(rest_of_X) else [])
                X_train = torch.cat(X_train_list, dim=0)
                X_train = X_train.to(device)
                y_train_list = rest_of_y[:start_index] + (rest_of_y[end_index:] if end_index < len(rest_of_y) else [])
                y_train = torch.cat(y_train_list, dim=0)

                # Train Indices
                idx_train_list = rest_indices[:start_index] + (rest_indices[end_index:] if end_index < len(rest_indices) else [])
                idx_train = torch.cat(idx_train_list, dim=0)

                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                test_dataset = TensorDataset(X_test, y_test)

                train_loader = DataLoader(
                    train_dataset,
                    sampler=ImbalancedDatasetSampler(train_dataset),
                    batch_size= batch_size
                )
                val_loader = DataLoader(val_dataset, batch_size = batch_size)
                test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
                model = d169_3d(num_classes=number_of_classes, sample_size=64, sample_duration=96, norm="bn", act="lrelu")
                model.to(device)

                # Model Config
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-10)
                
                ### Loggers for training process
                train_batch_log_path = Path(valfolddir + '/training_log_batch.csv')
                train_batch_log_header = ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr']
                train_batch_logger = Logger(train_batch_log_path, train_batch_log_header)

                train_epoch_log_path = Path(valfolddir + '/training_log_epoch.csv')
                train_epoch_log_header = ['epoch', 'loss', 'acc', 'lr']
                train_epoch_logger = Logger(train_epoch_log_path, train_epoch_log_header)

                val_log_path = Path(valfolddir + '/validation_log.csv')
                val_log_header = ['epoch', 'loss', 'acc', 'roc_auc', 'roc_prc']
                val_logger = Logger(val_log_path, val_log_header)
                
                val_confusion_log_path = Path(valfolddir + '/validation_confusion_log.csv')
                val_confusion_log_header = ['epoch','TN', 'FP', 'FN', 'TP']
                val_confusion_logger = Logger(val_confusion_log_path, val_confusion_log_header)

                ### Loggers for Evaluating the best model
                val_best_log_path = Path(valfolddir + '/validation_best_log.csv')
                val_best_log_header = ['loss', 'acc', 'roc_auc', 'roc_prc']
                val_best_logger = Logger(val_best_log_path, val_best_log_header)

                val_best_confusion_log_path = Path(valfolddir + '/validation_best_confusion_log.csv')
                val_best_confusion_log_header = ['TN', 'FP', 'FN', 'TP']
                val_best_confusion_logger = Logger(val_best_confusion_log_path, val_best_confusion_log_header)

                ### Loggers for Evaluating Probabilities on dataset
                train_val_log_path = Path(valfolddir + '/probability_extraction_log.csv')
                train_val_log_header = ['loss', 'acc', 'roc_auc', 'roc_prc']
                train_val_logger = Logger(train_val_log_path, train_val_log_header)

                train_val_confusion_log_path = Path(valfolddir + '/probability_extraction_confusion_log.csv')
                train_val_confusion_log_header = ['TN', 'FP', 'FN', 'TP']
                train_val_confusion_logger = Logger(train_val_confusion_log_path, train_val_confusion_log_header)

                test_log_path = Path(valfolddir + '/test_log.csv')
                test_log_header = ['loss', 'acc', 'roc_auc', 'roc_prc']
                test_logger = Logger(test_log_path, test_log_header)

                test_confusion_log_path = Path(valfolddir + '/test_confusion_log.csv')
                test_confusion_log_header = ['TN', 'FP', 'FN', 'TP']
                test_confusion_logger = Logger(test_confusion_log_path, test_confusion_log_header)

                bestmodel_path = Path(valfolddir + '/bestmodel.pt')
                earlystopper = EarlyStopping(model, earlystop_criterion, eval_start_epoch, patience, bestmodel_path, min_delta = min_delta)

                memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
                memory_allocated_gb = memory_allocated_bytes / (1024 ** 3)
                print(f" > GPU Memory Allocated: {memory_allocated_gb:.2f} GB")

                ######################################################################
                # TRAINING LOOP WITH EARLY STOPPING
                ######################################################################
                print(" > Training is getting started...")
                print(" > Current Fold : " + str(val_fold + 1) + " of 5")
                print(" > Training takes {} epochs at max.".format(num_epochs))
                for epoch in range(num_epochs):

                    train_loss = train_epoch(
                        epoch + 1, train_loader, model, criterion, optimizer, device, learning_rate, train_epoch_logger, train_batch_logger)

                    val_loss = val_epoch(epoch + 1, val_loader, model, criterion, device, val_logger, val_confusion_logger, earlystopper, earlystop_criterion)

                    print('Training Results for epoch: [{0}] complete'.format(epoch + 1))

                    if(earlystopper.should_I_Stop()):
                        print('Early Stopping.')
                        break

                model.load_state_dict(torch.load(bestmodel_path))
                filenames = torch.load('/mnt/project/intern3_data/FastSurfer/Extracted/_LABELTENSORS/filename_labels_nlr.pt')

                ######################################################################
                # EXTRACT PROBABILITIES FROM TRAIN & VALID SET
                ######################################################################
                train_val_dataset = TensorDataset(torch.cat([X_train, X_val], dim=0), torch.cat([y_train, y_val], dim=0))
                train_val_loader = DataLoader(train_val_dataset, batch_size = batch_size, shuffle=False)
                pred_probs_train_val = test_epoch(train_val_loader, model, criterion, device, train_val_logger, train_val_confusion_logger)
                best_val_probs = test_epoch(val_loader, model, criterion, device, val_best_logger, val_best_confusion_logger)

                filename_list_train_val = []
                LR_list_train_val = []
                pred_probs_0_list_train_val = []
                pred_probs_1_list_train_val = []

                train_val_indices = torch.cat([idx_train, idx_val], dim=0)
                if labels_nlr:
                    for (index, pred_prob) in zip(train_val_indices, pred_probs_train_val):
                        filename = filenames[index]
                        LR = '-'
                        filename_list_train_val.append(filename)
                        LR_list_train_val.append(LR)
                        pred_probs_0_list_train_val.append(pred_prob[0])
                        pred_probs_1_list_train_val.append(pred_prob[1])
                else:
                    for i, pred_prob in enumerate(pred_probs_train_val):
                        index = train_val_indices[i] % len(filenames)
                        filename = filenames[index]
                        LR = 'L' if train_val_indices[i] < len(filenames) else 'R'
                        filename_list_train_val.append(filename)
                        LR_list_train_val.append(LR)
                        pred_probs_0_list_train_val.append(pred_prob[0])
                        pred_probs_1_list_train_val.append(pred_prob[1])
                
                df_probs = pd.DataFrame({'Filename' : filename_list_train_val, 'LR' : LR_list_train_val, 'Prob_0' : pred_probs_0_list_train_val, 'Prob_1' : pred_probs_1_list_train_val})
                df_probs.to_csv(Path(valfolddir + '/train_probabilities_log.csv'), index=False)

                ######################################################################
                # TESTING
                ######################################################################
                pred_probs = test_epoch(test_loader, model, criterion, device, test_logger, test_confusion_logger)

                filename_list = []
                LR_list = []
                pred_probs_0_list = []
                pred_probs_1_list = []

                if labels_nlr:
                    for (index, pred_prob) in zip(idx_test, pred_probs):
                        filename = filenames[index]
                        LR = '-'
                        filename_list.append(filename)
                        LR_list.append(LR)
                        pred_probs_0_list.append(pred_prob[0])
                        pred_probs_1_list.append(pred_prob[1])
                else:
                    for i, pred_prob in enumerate(pred_probs):
                        index = idx_test[i] % len(filenames)
                        filename = filenames[index]
                        LR = 'L' if idx_test[i] < len(filenames) else 'R'
                        filename_list.append(filename)
                        LR_list.append(LR)
                        pred_probs_0_list.append(pred_prob[0])
                        pred_probs_1_list.append(pred_prob[1])

                df_probs = pd.DataFrame({'Filename' : filename_list, 'LR' : LR_list, 'Prob_0' : pred_probs_0_list, 'Prob_1' : pred_probs_1_list})
                df_probs.to_csv(Path(valfolddir + '/test_probabilities_log.csv'), index=False)