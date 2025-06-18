import tensorflow as tf
import numpy as np
from sklearn import preprocessing

raw_csv=np.loadtxt('Audiobooks_data.csv', delimiter=',')

#BALANCING THE DATA
unscaled_inputs=raw_csv[:,1:-1]
targets_all=raw_csv[:,-1]

num_one_target=int(np.sum(targets_all))
num_zero_target=0
indices_to_delete=[]

for i in range(targets_all.shape[0]):
    if targets_all[i]==0:
        num_zero_target +=1
    if num_zero_target > num_one_target:
        indices_to_delete.append(i)

unscaled_balanced_inputs=np.delete(unscaled_inputs, indices_to_delete, axis=0)
balanced_targets=np.delete(targets_all, indices_to_delete, axis=0)

#Scale the data with the preprocessing module
scaled_inputs=preprocessing.scale(unscaled_balanced_inputs)

#SHUFFLE THE DATA
BATCH_SIZE=100
shuffled_indicies=np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indicies)

shuffled_inputs=scaled_inputs[shuffled_indicies]
shuffled_targets=balanced_targets[shuffled_indicies]

#SPLIT THE DATASET INTO TRAIN, TEST AND VALIDATE
size_of_dataset=shuffled_inputs.shape[0]
num_validate= int(size_of_dataset *0.1)
num_train=int(size_of_dataset * 0.8)
num_test=size_of_dataset - num_validate - num_train 

train_inputs= shuffled_inputs[:num_train]
train_target=shuffled_targets[:num_train]

validate_inputs= shuffled_inputs[num_train: num_train + num_validate]
validate_target= shuffled_targets[num_train: num_train + num_validate]
 
test_inputs=shuffled_inputs[num_validate + num_train:]
test_targets= shuffled_targets[num_validate + num_train:]

#print(np.sum(train_target), num_train, np.sum(train_target)/num_train)
#print(np.sum(validate_target), num_validate, np.sum(validate_target)/num_validate)
#print(np.sum(test_targets), num_test, np.sum(test_targets)/num_test)

#SAVE THE THREE FILES IN NPX 
np.savez('Audiobooks_data_train', inputs=train_inputs, targets= train_target)
np.savez('Audiobooks_data_validation', inputs=validate_inputs, targets=validate_target)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)