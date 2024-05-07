######################################################################
########################## CONFIGURATIONS ############################
######################################################################
random_state = 77777
######################################################################
pathogenlist = [0, 1, 2, 3]
# 0 = autoimmune
# 1 = bacterial
# 2 = tuberculosis
# 3 = viral
######################################################################
labels_to_classify = [3, 4, 5, 6] # The labels with smaller quantity
# Model Hyperparameters
weight_decay = 0.1
num_epochs = 1000
learning_rate = 0.00005
number_of_classes = 2
batch_size = 4
# Early stopping Config
patience = 10
eval_start_epoch = 40
earlystop_criterion = 'AUROC'
min_delta = 0.0005
n_input_channels = 1
num_test_folds = 5 # num of splits
test_fold = 1 # idx of fold to use as internal test set
num_val_folds = 4 # num of validation folds
idxnum = 1 # num of indices per val. fold
# Directories
homedir = '/home/yhchoi/24CNSInfectionMRI'
y_dir = '/home/yhchoi/24CNSInfectionMRI/Labels/prognosis_labels_internal.pt'
tensordir_dir = '/home/yhchoi/24CNSInfectionMRI/Results/1_Outputdir'
filenames_dir = '/home/yhchoi/24CNSInfectionMRI/Labels/filename_labels_internal.pt'
#TODO : tidy up directories
num_of_files = 413 # Needed for efficient index counting (no need to call len() every time we need the number of files)
######################################################################