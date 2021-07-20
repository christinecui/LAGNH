#!/bin/bash
set -e

for i in 16 32 64 128
do
  echo "Start GAEH algorithm"
  CUDA_VISIBLE_DEVICES=0 python GAEH.py --nbits $i --beta 100 --lamda 1 --alpha 1 --nbits $i --dataset COCO --n_class 80 --hidden1 1024  --batch_size 1024 --epochs 300 --hidden2 $i
  echo "End ARGA algorithm"
  cd ../matlab
  matlab -nojvm -nodesktop -r "test_save($i, 'COCO', 'test'); quit;"
  cd ../ARGA
done