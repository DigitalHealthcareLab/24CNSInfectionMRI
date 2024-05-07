import dicom2nifti as d2n
import os
from pathlib import Path
from tqdm import tqdm

# Path containing all DICOM files
dir = Path('Your_DICOM_dir')
# Output Directory
opdir = Path('Yout_Output_Dir')

os.makedirs(opdir, exist_ok=True)

# Convert all DICOM files in the directory to NIFTI files
with tqdm(dir.iterdir()) as pbar:
    for subdir in pbar:
        if subdir.is_dir():
            # Convert all DICOM files in the subdirectory to NIFTI files
            dirname = str(subdir).split('/')[-1]
            os.mkdir(f'{opdir}/{dirname}')
            pbar.set_description(f'Processing {dirname}')
            d2n.convert_directory(subdir, f'{opdir}/{dirname}', compression=True, reorient=True)