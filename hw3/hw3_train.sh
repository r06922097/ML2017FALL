#!/bin/bash
python3 mnist.py "$1"
python3 train2.py "$1"
python3 train3.py "$1"
python3 train_TA.py "$1"
python3 train_40x40.py "$1"

