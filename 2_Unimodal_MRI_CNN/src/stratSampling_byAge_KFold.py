import torch

def add_and_shuffle(y1, y2, y3, random_state = 42):
    torch.manual_seed(random_state)
    y_temp = torch.cat([y1, y2, y3], dim=0)
    
    # indices = torch.randperm(len(y_temp)).tolist()
    # y = y_temp[indices]
    # return y
    return y_temp

def split_5(indices, random_state=42):
    torch.manual_seed(random_state)
    split_size = len(indices) // 5
    remainder = len(indices) % 5

    folds = []
    start_idx = 0

    for i in range(5):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        folds.append(indices[start_idx:end_idx])
        start_idx = end_idx

    return folds[0], folds[1], folds[2], folds[3], folds[4]

def Strat_5fold_Split_Indices_ByAge(indices, Ages, random_state = 42):
    torch.manual_seed(random_state)
    age_groups = torch.zeros_like(Ages)

    age_groups[Ages < 40] = 1
    age_groups[(Ages >= 40) & (Ages < 60)] = 2
    age_groups[Ages >= 60] = 3

    ind_young = torch.where(age_groups == 1)[0]
    ind_middle = torch.where(age_groups == 2)[0]
    ind_old = torch.where(age_groups == 3)[0]

    y_young = indices[ind_young]
    y_middle = indices[ind_middle]
    y_old = indices[ind_old]

    fold0y, fold1y, fold2y, fold3y, fold4y = split_5(y_young, random_state=random_state)
    fold0m, fold1m, fold2m, fold3m, fold4m = split_5(y_middle, random_state=random_state)
    fold0o, fold1o, fold2o, fold3o, fold4o = split_5(y_old, random_state=random_state)

    (lambda b : print([len(b[0][i]) for i in range(5)]))([split_5(y_young, random_state=random_state)])
    (lambda b : print([len(b[0][i]) for i in range(5)]))([split_5(y_middle, random_state=random_state)])
    (lambda b : print([len(b[0][i]) for i in range(5)]))([split_5(y_old, random_state=random_state)])

    fold0 = add_and_shuffle(fold0y, fold0m, fold0o, random_state=random_state)
    fold1 = add_and_shuffle(fold1y, fold1m, fold1o, random_state=random_state)
    fold2 = add_and_shuffle(fold2y, fold2m, fold2o, random_state=random_state)
    fold3 = add_and_shuffle(fold3y, fold3m, fold3o, random_state=random_state)
    fold4 = add_and_shuffle(fold4y, fold4m, fold4o, random_state=random_state)

    return [fold0, fold1, fold2, fold3, fold4]
    