import numpy as np
import glob
import cv2
import os

import SudokuRecognizer as sr
from mnist import MNIST

# ADDITIONAL LIBRARIES
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import Sequential

import matplotlib.pyplot as plt
import seaborn as sns
import time

#Define your functions here if required :

# ----- Get input data from MNIS dataset ----
def input_data(train_images, train_labels,test_images,test_labels):


    X_train = np.array(list(train_images)).astype(np.float32)
    X_test = np.array(list(test_images)).astype(np.float32)

    y_train = np.array(list(train_labels)).astype(np.float32)
    y_test = np.array(list(test_labels)).astype(np.float32)


    # ----- Reshape to be samples*pixels*width*height -----
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # ----- One hot -----
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # ----- Convert from integers to floats -----
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # ----- Normalize to range [0, 1] -----
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    return X_test, y_test, X_train, y_train

# ----- Create model function and building CNN -----
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # ----- Compile model -----
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ----- Create test function for MNIST dataset testing
def test(X_train, model):
    test_images = X_train[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        prediction = model.predict_classes(test_image, verbose=0)

        print("Predicted digit: {}".format(prediction[0]))
        plt.subplot(220 + i)
        plt.axis('off')
        plt.title("Predicted digit: {}".format(prediction[0]))
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))


    plt.show()


# ----- Create train funtion for MNIST -----
def train_run(train_images, train_labels,test_images,test_labels):

    X_test, y_test, X_train, y_train = input_data(train_images, train_labels,test_images,test_labels)

    model = create_model()

    # ----- Fitting the epochs and batch size for training data -----
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

    # ----- Testing the model -----
    test(X_train, model)

    # ----- Serialize model to JSON -----
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # ----- Saving the weights -----
    model.save_weights("model.hdf5")
    print("Saved model to disk")

    # ----- Confisuon Matrix for MNIST -----
    prediction = model.predict_classes(X_test, verbose=0)
    Y_true = np.argmax(y_test, axis=1)
    confusion_mtx = confusion_matrix(Y_true, prediction)
    sns.heatmap(confusion_mtx, annot=True, fmt="d")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


# Set dataset path before start example  '/Home/sudoku_dataset-master' :
sudoku_dataset_dir = '/Users/haruntanriverdi/Downloads/sudoku_dataset-master'
MNIST_dataset_dir = "data"

# ----- Load MNIST Dataset: -----
mndata = MNIST(MNIST_dataset_dir)
train_images, train_labels = mndata.load_training()
test_images,  test_labels = mndata.load_testing()

# ---- Initialize the training function -----
print("Training MNIST Dataset...")
time.sleep(2)
train_run(train_images, train_labels,test_images,test_labels)

# Apply PCA to MNIST :
# use sr.mnistPCA() that you applier for transformation
# classify test set with any method you choose (hint simplest one : nearest neighbour)
# report the outcome
# Calculate confusion matrix, false postivies/negatives.
# print(reporting_results)


image_dirs = sudoku_dataset_dir + '/images/image*.jpg'
data_dirs = sudoku_dataset_dir + '/images/image*.dat'
IMAGE_DIRS = glob.glob(image_dirs)
IMAGE_DIRS.sort()
DATA_DIRS = glob.glob(data_dirs)
DATA_DIRS.sort()
len(IMAGE_DIRS)


#Define your variables etc. outside for loop here:

# Accumulate accuracy for average accuracy calculation.
cumulativeAcc = 0

## Loop over all images
for img_dir, data_dir in zip(IMAGE_DIRS, DATA_DIRS):

    #Define your variables etc.:
    image_name = os.path.basename(img_dir)
    data = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
    img = cv2.imread(img_dir)

    # detect sudoku puzzle:
    warp = sr.detectSudoku(img)

    # Recognize digits in sudoku puzzle :
    sudokuArray = sr.RecognizeSudoku(warp)

    #print(sudokuArray)
    # cv2.imshow('cell', warp)  # Display the image
    # cv2.waitKey(0)

    # Evaluate Result for current image :

    detectionAccuracyArray = data == sudokuArray
    accPercentage = np.sum(detectionAccuracyArray)/detectionAccuracyArray.size
    cumulativeAcc = cumulativeAcc + accPercentage
    print(image_name + " accuracy : " + accPercentage.__str__() + "%")

# Average accuracy over all images in the dataset :
averageAcc = cumulativeAcc/len(IMAGE_DIRS)
print("dataset performance : " + averageAcc.__str__() + "%")


