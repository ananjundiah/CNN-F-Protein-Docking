#!/bin/bash

python train_protein.py --max-cycles 1 \
                --path_train /central/groups/smayo/avinashsLargeStorage/PDB_NR_AceMD_FullSet/Contact_Maps/Final_Set_600/Train/ \
                --path_test /central/groups/smayo/avinashsLargeStorage/PDB_NR_AceMD_FullSet/Contact_Maps/Final_Set_600/Test/ \
                --ind 0 \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --clean 'supclean' \
                --clean-parameter 0.1 \
                --lr 0.05 \
                --batch-size 4 \
                --eps 0.2 \
                --eps-iter 0.071 \
                --schedule 'poly' \
                --epochs 200 \
                --seed 1 \
                --grad-clip \
                --save-model 'CNNF_0_protein_contact' \
                --model-dir 'models_protein_contact'


wait 
echo "All done"
