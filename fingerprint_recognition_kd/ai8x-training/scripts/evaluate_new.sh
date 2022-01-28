#!/bin/sh
./train.py --model ai85net_fpr --dataset fpr --evaluate --exp-load-weights-from ../ai8x-synthesis/proj/qat_best.pth.tar -8 --device MAX78000 "$@"
