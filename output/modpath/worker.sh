#!/bin/sh
tar xzf data.tar
cd data

export PATH=$PWD/neversinkrun/bin:$PATH
echo $PATH

# the only argument passed is the realization number for MC
realnum=$1

python modpath_mc.py $realnum