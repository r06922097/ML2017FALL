#!/bin/bash
wget -O model_train5_6.h5 https://www.dropbox.com/s/b8xdr83zskcr5q3/model_train5_6.h5?dl=1
python3 predict.py "$1" "$2" "$3" "$4" 
