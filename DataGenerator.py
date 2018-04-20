## https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
import keras
from Encoding_in_chunks import OneHotEncoder

# Let ID be the Python string that identifies a given sample of the dataset. A good way to keep track of samples and their labels is #  to adopt the following framework:

# Create a dictionary called partition where you gather:
#    1. partition['train'] a list of training IDs
#    2. partition['validation'] a list of validation IDs

# Create a dictionary called labels where for each ID of the dataset, the associated label is given by labels[ID]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        print(X.shape, "#######")

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size, self.dim))

        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('../input/train_' + str(ID) + '.npy')

            # Store class
            y[i,] = np.load('../input/target_'+str(ID)+'.npy')
            
            
        if self.n_classes > 0:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X, y