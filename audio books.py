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

#LOAD THE PREPROCESSED DATA
npz_train=np.load('Audiobooks_data_train.npz')
training_inputs=npz_train['inputs'].astype(np.float32)
training_targets=npz_train['targets'].astype(np.int32)


npz_validate=np.load('Audiobooks_data_validation.npz')
validation_inputs=npz_validate['inputs'].astype(np.float32)
validation_targets=npz_validate['targets'].astype(np.int32)


npz_test=np.load('Audiobooks_data_validation.npz')
testing_inputs=npz_test['inputs'].astype(np.float32)
testing_targets=npz_test['targets'].astype(np.int32)


#OUtlining the model
input_size=10
hidden_layer=100
output_size=2

model=tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer, activation='relu'),
    tf.keras.layers.Dense(hidden_layer, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

#OPTIMIZER AND LOSS
#CREATE AN EARLY STOPPING MECHANISM BECAUSE OF OVERFITTING NOTICED
lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.9
)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
batch_size=100
early_stopping=tf.keras.callbacks.EarlyStopping(patience=2)
max_epochs=100
model.fit(training_inputs, training_targets,
           batch_size=batch_size, epochs= max_epochs,
           callbacks=[early_stopping],
           validation_data=(validation_inputs, validation_targets) ,
           verbose=2)

#Test the data
test_loss, test_accuracy=model.evaluate(training_inputs, training_targets)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss,test_accuracy*100))

