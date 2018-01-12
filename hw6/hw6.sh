#!/bin/bash
wget -O autoencoder_96402.h5 https://www.dropbox.com/s/fserwo2o3qs1req/autoencoder_96402.h5?dl=0
wget -O encoder_96402.h5 https://www.dropbox.com/s/mp5maibnqe4lqm4/encoder_96402.h5?dl=1
wget -O decoder_96402.h5 https://www.dropbox.com/s/gyt599ghlc3gw09/decoder_96402.h5?dl=1
python3 predict.py "$1" "$2" "$3"