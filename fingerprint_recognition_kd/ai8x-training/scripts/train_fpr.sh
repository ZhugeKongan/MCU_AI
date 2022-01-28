#!/bin/sh
#python train.py --optimizer Adam --lr 0.001 --batch-size 64 --gpus 0 --epochs 160 --deterministic --compress schedule.yaml --model ai85fprnet --dataset fpr --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"
python train.py --gpus 0 --epochs 160 --optimizer Adam --lr 0.001 --deterministic --compress schedule-fpr.yaml --model ai85fprnet --dataset fpr --batch-size 64 --device MAX78000 --regression --print-freq 250 "$@"
#--sense channel