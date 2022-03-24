How to read data.

First look at dat2mat.m and parse_aedat.m
Then, do the following

====================================================================================================
find ABSOLUTE_PATH_TO_CIFAR_DVS -type f | grep aedat | sort > aedat_list.txt
cd PATH_TO_CIFAR_DVS_ROOT
mkdir mat_data
cd data
find . -type d > dirs.txt
cd ../mat_data
xargs mkdir -p < dirs.txt
====================================================================================================

Go to MATLAB, and run the following

====================================================================================================
parse_aedat(PATH_TO_AEDAT_LIST)
====================================================================================================

To check progress, run the following

====================================================================================================
find mat_data -type f | grep aedat | wc -l
====================================================================================================
