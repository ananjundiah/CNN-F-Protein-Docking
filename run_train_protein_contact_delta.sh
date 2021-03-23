#!/bin/bash

python train_protein_delta.py --max-cycles 1 \
                --path_train /home/ubuntu/Final_Set_600/Train/ \
                --path_test /home/ubuntu/Final_Set_600/Test/ \
                --ind 0 \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --lr 0.05 \
                --batch-size 4 \
                --eps 0.2 \
                --eps-iter 0.071 \
                --schedule 'poly' \
                --epochs 200 \
                --seed 1 \
                --grad-clip \
                --save-model 'CNNF_0_protein_contact_delta' \
                --model-dir 'models_protein_contact' > CNNF_0_protein_contact_delta.out


wait 
echo "All done"
