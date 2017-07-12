# ~~> Import data

import csv

# Read CSVfile of images
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
print ('Number of lines read in csv: ' + str(len(lines)))

# ~~> Construct list of images and steering directions

# Construct list containing image paths and steering directions
samples = []
for line in lines:
    
    # Contruct of centre images and steering directions
    source_path = line[0]
    filename = source_path.split('/')[-1]
    relative_path = './data/IMG/'
    current_path = relative_path + filename
    measurement = float(line[3])
    samples.append([current_path, measurement, ''])

    # Use side images with corrected steering measurement
    correction = 0.1
    measurement_left = measurement + correction
    measurement_right = measurement - correction
    left_image_path = relative_path + 'left' + filename[6:]
    right_image_path = relative_path + 'right' + filename[6:]
    samples.append([left_image_path,measurement_left,''])
    samples.append([right_image_path,measurement_right,''])

# Augment training set by flipping each image horizontally
# and reversing the steering angle
flipped_samples = [[x[0],x[1] * -1.0, 'flip'] for x in samples]
samples = samples + flipped_samples

# ~~> Create training and validation sets

from sklearn.utils import shuffle

# Create random sample of training (80%) and validation (20%) sets
samples = shuffle(samples)

num_validation_set = len(samples) // 5
validation_set = samples[:num_validation_set]
training_set = samples[num_validation_set:]

print ('Number of samples: ' + str(len(samples)))
print ('Size of training set: ' + str(len(training_set)))
print ('Size of validation set: ' + str(len(validation_set)))

# ~~> Generator for images

import cv2
import numpy as np

BATCH_SIZE = 32 # number of lines in a single batch
num_training_examples = len(samples)

# A generator to output batches of samples and steering directions
# which can fit into memory
def generate_training_batch(training_set):
    
    while True:
        
        # Before each epoch, shuffle the training set
        # (shuffling should not have an impact on validation set)
        training_set = shuffle(training_set)
        
        # From training set, create a batch of size BATCH_SIZE
        for offset in range(0,len(training_set),BATCH_SIZE):
            end = offset + BATCH_SIZE
            
            images = []
            measurements = []
            
            # Process BATCH_SIZE number of lines in training set
            for example in training_set[offset:end]:
                
                image = cv2.imread(example[0])

                # Augment training set by flipping images
                if (example[2] == 'flip'):
                    image = cv2.flip(image,1)
                
                images.append(image)
                measurement = float(example[1])
                measurements.append(measurement)
            
            # Convert images and steering measurements to Numpy arrays
            # and yield (X_train, y_train)
            yield (np.array(images), np.array(measurements))

# Create training and validation generators
training_batch_generator = generate_training_batch(training_set)
validation_batch_generator = generate_training_batch(validation_set)


# ~~> Load existing model and train
# If builidng upon existing model uncomment the code below. This will 
# continue training with newly collected training data. 

# from keras.models import Sequential, load_model

# Load previously saved model and training using previous weights
# model = load_model('model - in.h5')
# history_object = model.fit_generator(generator=training_batch_generator, samples_per_epoch=len(training_set), nb_epoch=5,validation_data=validation_batch_generator, nb_val_samples=len(validation_set))
# model.save('model - out.h5')


# ~~> Construct neural network architecture

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Activation, MaxPooling2D, Dropout, Cropping2D

dropout_prob = 0.2
model = Sequential()

# Normalise and mean centre the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop top of image (containing trees, etc) and bottom (hood of car)
model.add(Cropping2D(cropping = ((70,25),(0,0))))

# Following based on NVidia architecture with drop-out layers to prevent overfitting
model.add(Convolution2D(24,5,5,border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5,border_mode='valid', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(dropout_prob))
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))


# ~~> Compile and train model, then save output

model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(generator=training_batch_generator, samples_per_epoch=len(training_set), nb_epoch=5,validation_data=validation_batch_generator, nb_val_samples=len(validation_set))
model.save('model - out.h5')






