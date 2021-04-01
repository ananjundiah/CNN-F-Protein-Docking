import os

batches = [4,6,8,10]
lrs = [0.05, 0.005, 0.0005]
fl = open('run_train_protein_contact_delta.sh').readlines()

batch = 6
lr = 0.0005

for batch in batches:
    for lr in lrs:
        fl[9] = '                --lr ' + str(lr) + ' \\\n'
        fl[10] = '                --batch-size ' + str(batch) + ' \\\n'
        fl[17] = "                --save-model 'batch_" + str(batch) + "_lr_" + str(lr) + "' \\\n"
        fl[18] = "                --model-dir 'models_batch_lr_tune' > logs_batch_lr_tune/batch_" + str(batch) + "_lr_" + str(lr) + ".out\n"
        out = open('run_train_tune.sh', 'w')
        out.write(''.join(fl))
        out.close()
        os.system('bash run_train_tune.sh')


