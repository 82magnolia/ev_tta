How to read data.

First look at dat2mat.m and parse_dat.m
Then, do the following

====================================================================================================
find ABSOLUTE_PATH_TO_N-CARS -type f | grep dat | sort > dat_list.txt
cd PATH_TO_N-CARS_ROOT
mkdir mat_data
cd data
find . -type d > dirs.txt
cd ../mat_data
xargs mkdir -p < dirs.txt
====================================================================================================

Go to MATLAB, and run the following

====================================================================================================
parse_dat(PATH_TO_DAT_LIST)
====================================================================================================

To check progress, run the following

====================================================================================================
find mat_data -type f | grep dat | wc -l
====================================================================================================
