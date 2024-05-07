import torch

def pathogen_extraction(X, y, age, pathogens, pathogen_to_extract):
    pathogen_groups = torch.zeros_like(pathogens)

    pathogen_groups[pathogens == pathogen_to_extract] = 1
    pathogen_groups[pathogens != pathogen_to_extract] = 0

    indices_toextract = torch.where(pathogen_groups == 1)[0]

    X_extracted = X[indices_toextract]
    y_extracted = y[indices_toextract]
    age_extracted = age[indices_toextract]

    return X_extracted, y_extracted, age_extracted