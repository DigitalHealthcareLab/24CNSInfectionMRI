# labels1 = [
# 'Cerebellum-White-Matter',
# 'Cerebellum-Cortex',
# 'caudalanteriorcingulate',
# 'caudalmiddlefrontal'
# ]

# labels2 = [
# 'cuneus',
# 'entorhinal',
# 'fusiform',
# ] # Done

# labels3 = [
# 'inferiortemporal', 
# 'isthmuscingulate',# Done
# 'lateraloccipital',
# 'lateralorbitofrontal'
# ]

# labels4 = [
# 'lingual',
# 'medialorbitofrontal',
# 'parahippocampal',
# 'paracentral'
# ] # Done

# labels5 = [
# 'parsopercularis',
# 'parsorbitalis',
# 'parstriangularis',
# 'pericalcarine'
# ] # Done

# labels6 = [
# 'posteriorcingulate',
# 'precuneus',
# 'rostralanteriorcingulate',
# 'rostralmiddlefrontal'
# ] # Done

# labels7 = [
# 'superiorfrontal',
# 'superiortemporal',
# 'superiorparietal'
# ]

# labels8 = [
# 'supramarginal',
# 'transversetemporal',
# 'insula'
# ] # Done

# labels9 = [
# 'postcentral',
# 'inferiorparietal'
# ]

# #-----------------------------#
# # Done

# labels10 = [
# 'precentral' 
# ]

# labels11 = [
#   'middletemporal' # TODO
# ]

# # Complete
# inner0 = ['Thalamus-Proper']
# inner1 = ['Caudate']
# inner2 = ['Putamen']
# inner3 = ['Pallidum']
# inner4 = ['Brain-Stem']
# inner5 = ['Hippocampus']
# inner6 = ['Amygdala']
# inner7 = ['Accumbens-area']
# inner8 = ['VentralDC']
# inner9 = ['Choroid-plexus']

#-----------------------------#

# labels dict : mapping names to values in the segmentation mask
labels_lookup = {
'Left-Lateral-Ventricle': 4,
'Left-Inf-Lat-Vent': 5,
'Left-Cerebellum-White-Matter': 7,
'Left-Cerebellum-Cortex': 8,
'Left-Thalamus-Proper': 10,
'Left-Caudate': 11,
'Left-Putamen': 12,
'Left-Pallidum': 13,
'Left-3rd-Ventricle': 14,
'Left-4th-Ventricle': 15,
'Left-Brain-Stem': 16,
'Left-Hippocampus': 17,
'Left-Amygdala': 18,
'Left-CSF': 24,
'Left-Accumbens-area': 26,
'Left-VentralDC': 28,
'Left-Choroid-plexus': 31,
'Right-Lateral-Ventricle': 43,
'Right-Inf-Lat-Vent': 44,
'Right-Cerebellum-White-Matter': 46,
'Right-Cerebellum-Cortex': 47,
'Right-Thalamus-Proper': 49,
'Right-Caudate': 50,
'Right-Putamen': 51,
'Right-Pallidum': 52,
'Right-Hippocampus': 53,
'Right-Amygdala': 54,
'Right-Accumbens-area': 58,
'Right-VentralDC': 60,
'Right-Choroid-plexus': 63,
'Right-3rd-Ventricle': 14,
'Right-4th-Ventricle': 15,
'Right-Brain-Stem': 16,
'Right-CSF': 24,
'ctx-lh-caudalanteriorcingulate': 1002,
'ctx-lh-caudalmiddlefrontal': 1003,
'ctx-lh-cuneus': 1005,
'ctx-lh-entorhinal': 1006,
'ctx-lh-fusiform': 1007,
'ctx-lh-inferiorparietal': 1008,
'ctx-lh-inferiortemporal': 1009,
'ctx-lh-isthmuscingulate': 1010,
'ctx-lh-lateraloccipital': 1011,
'ctx-lh-lateralorbitofrontal': 1012,
'ctx-lh-lingual': 1013,
'ctx-lh-medialorbitofrontal': 1014,
'ctx-lh-middletemporal': 1015,
'ctx-lh-parahippocampal': 1016,
'ctx-lh-paracentral': 1017,
'ctx-lh-parsopercularis': 1018,
'ctx-lh-parsorbitalis': 1019,
'ctx-lh-parstriangularis': 1020,
'ctx-lh-pericalcarine': 1021,
'ctx-lh-postcentral': 1022,
'ctx-lh-posteriorcingulate': 1023,
'ctx-lh-precentral': 1024,
'ctx-lh-precuneus': 1025,
'ctx-lh-rostralanteriorcingulate': 1026,
'ctx-lh-rostralmiddlefrontal': 1027,
'ctx-lh-superiorfrontal': 1028,
'ctx-lh-superiorparietal': 1029,
'ctx-lh-superiortemporal': 1030,
'ctx-lh-supramarginal': 1031,
'ctx-lh-transversetemporal': 1034,
'ctx-lh-insula': 1035,
'ctx-rh-caudalanteriorcingulate': 2002,
'ctx-rh-caudalmiddlefrontal': 2003,
'ctx-rh-cuneus': 2005,
'ctx-rh-entorhinal': 2006,
'ctx-rh-fusiform': 2007,
'ctx-rh-inferiorparietal': 2008,
'ctx-rh-inferiortemporal': 2009,
'ctx-rh-isthmuscingulate': 2010,
'ctx-rh-lateraloccipital': 2011,
'ctx-rh-lateralorbitofrontal': 2012,
'ctx-rh-lingual': 2013,
'ctx-rh-medialorbitofrontal': 2014,
'ctx-rh-middletemporal': 2015,
'ctx-rh-parahippocampal': 2016,
'ctx-rh-paracentral': 2017,
'ctx-rh-parsopercularis': 2018,
'ctx-rh-parsorbitalis': 2019,
'ctx-rh-parstriangularis': 2020,
'ctx-rh-pericalcarine': 2021,
'ctx-rh-postcentral': 2022,
'ctx-rh-posteriorcingulate': 2023,
'ctx-rh-precentral': 2024,
'ctx-rh-precuneus': 2025,
'ctx-rh-rostralanteriorcingulate': 2026,
'ctx-rh-rostralmiddlefrontal': 2027,
'ctx-rh-superiorfrontal': 2028,
'ctx-rh-superiorparietal': 2029,
'ctx-rh-superiortemporal': 2030,
'ctx-rh-supramarginal': 2031,
'ctx-rh-transversetemporal': 2034,
'ctx-rh-insula': 2035}

# Function for finding labels
def label_lookups(structure, hemi):
  # determine what to plot
  if structure[0].isupper() or structure.startswith(('3', '4')):
    if hemi == "left":
      label = labels_lookup["Left-" + structure]
    elif hemi == "right":
      label = labels_lookup["Right-" + structure]
    else:
      label = [labels_lookup["Left-" + structure], labels_lookup["Right-" + structure]]
  else:
    if hemi == "left":
      label = labels_lookup["ctx-lh-" + structure]
    elif hemi == "right":
      label = labels_lookup["ctx-rh-" + structure]
    else:
      label = [labels_lookup["ctx-lh-" + structure], labels_lookup["ctx-rh-" + structure]]
  return label

# All labels
labels = [
# Unwanted labels
# 'Lateral-Ventricle',  
# 'Inf-Lat-Vent',
# '3rd-Ventricle',
# '4th-Ventricle',
# 'CSF',
  
# Cerebellum areas
'Cerebellum-White-Matter',
'Cerebellum-Cortex',

# Limbic areas
'Thalamus-Proper',
'Caudate',
'Putamen',
'Pallidum',
'Brain-Stem',
'Hippocampus',
'Amygdala',
'Accumbens-area',
'VentralDC',
'Choroid-plexus',

# Cortical areas
'caudalanteriorcingulate',
'caudalmiddlefrontal',
'cuneus',
'entorhinal',
'fusiform',
'inferiorparietal',
'inferiortemporal', 
'isthmuscingulate',
'lateraloccipital',
'lateralorbitofrontal',
'lingual',
'medialorbitofrontal',
'middletemporal',
'parahippocampal',
'paracentral',
'parsopercularis',
'parsorbitalis',
'parstriangularis',
'pericalcarine',
'postcentral',
'posteriorcingulate',
'precentral',
'precuneus',
'rostralanteriorcingulate',
'rostralmiddlefrontal',
'superiorfrontal',
'superiorparietal',
'superiortemporal',
'supramarginal',
'transversetemporal',
'insula'
]