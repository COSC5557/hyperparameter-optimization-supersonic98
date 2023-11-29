import math
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import layers
import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def train_test_split(input, output, r = 0.2):
    N_images = input.shape[0]

    r = 1 - r
    # x train split
    x_train = input[0:math.floor(N_images*r), :, :]

    # x test split
    x_test = input[math.floor(N_images * r):N_images, :, :]

    # y train split
    y_train = output[0:math.floor(N_images*r), :, :]

    # y test split
    y_test = output[math.floor(N_images * r):N_images, :, :]

    return x_train, y_train, x_test, y_test


def max_normalize(data):
    # apply max normalization
    n_samples = data.shape[0]
    for i in range(0, n_samples):
        data[i, :, :] = data[i, :, :] / np.max(data[i, :, :])

    #
    return data


def remove_mean(data):
    # subtract mean from every image
    n_samples = data.shape[0]
    for i in range(0, n_samples):
        data[i, :, :] = data[i, :, :]  - np.mean(data[i, :, :])

    return data


def plot_images(imgs1, imgs2, labels):

    n_images = imgs1.shape[0]
    assert imgs1.shape[0] == imgs2.shape[0], "number of imgs 1 and 2 are not equal"

    fig, axes = plt.subplots(n_images, 2)

    for i in range(n_images):
        im = axes[i, 0].imshow(imgs1[i], cmap='seismic')
        axes[i, 0].axis('off')
        fig.colorbar(im, ax=axes[i, 0])
        axes[i, 0].set_title(labels[0])

        im = axes[i, 1].imshow(imgs2[i], cmap='seismic')
        axes[i, 1].axis('off')
        fig.colorbar(im, ax=axes[i, 1])
        axes[i, 1].set_title(labels[1])


    plt.tight_layout()
    plt.show()


def build_cnn_model(settings, input_shape=(500, 500, 1)):
    activation = settings[0]
    loss = settings[1]
    optimizer = settings[2]

    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(64, (3, 3), activation = activation, input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation=activation, padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), activation=activation, padding='same'))
    model.add(MaxPooling2D((5, 5), padding='same'))

    # Up-sampling layers
    model.add(Conv2D(256, (3, 3), activation=activation, padding='same'))
    model.add(UpSampling2D((5, 5)))

    model.add(Conv2D(128, (3, 3), activation=activation, padding='same'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation=activation, padding='same'))
    model.add(UpSampling2D((2, 2)))

    # Output layer
    model.add(Conv2D(1, (3, 3), activation=activation, padding='same'))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)

    return model


def train_model(model, x_train, y_train, epochs, batch_size, cv_folds = 5):
    n_samples = x_train.shape[0]
    epochs = math.floor(epochs / cv_folds)
    kf = KFold(n_splits=cv_folds)
    loss = []
    val_loss = []
    for iter in range(epochs):
        print(f"Iteration {iter}:")
        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            print(f"Fold {i}:")
            x = x_train[train_index]
            y = y_train[train_index]
            x_val = x_train[test_index]
            y_val = y_train[test_index]
            h = model.fit(x, y, batch_size=batch_size, epochs=1, verbose=1, validation_data = (x_val, y_val), shuffle=True)
            loss.append(h.history['loss'])
            val_loss.append(h.history['val_loss'])
    return model, loss, val_loss


def hpo_grid_search(x_train, y_train, x_test, y_test):
    # Define search space
    activations  = ['relu', 'sigmoid', 'tanh', 'softmax']
    losses = ['mean_squared_error', 'mean_absolute_error',  'mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity']
    optimizers = ['RMSprop' ,'adam', 'adamax', 'Adadelta']
    n_cases = len(activations) * len(losses) * len(optimizers)

    tuning_results = dict()
    tuning_results['activation'] = []
    tuning_results['loss function'] = []
    tuning_results['optimizer'] = []
    tuning_results['loss curve'] = []
    tuning_results['val loss'] = []
    tuning_results['training error'] = []
    tuning_results['test error'] = []
    tuning_results['models'] = []

    # Search
    for act in activations:
        for loss in losses:
            for opt in optimizers:
                settings = [act, loss, opt]
                print("Training with settings: ", settings)
                model = build_cnn_model(settings)
                model, loss_curve, val_loss = train_model(model, x_train, y_train, batch_size = 1, epochs = 5)

                tuning_results['models'].append(model)
                tuning_results['activation'].append(act)
                tuning_results['loss function'].append(loss)
                tuning_results['optimizer'].append(opt)
                tuning_results['loss curve'].append(loss_curve)
                tuning_results['val loss'].append(val_loss)

                # evaluate performance
                train_err = model.evaluate(x_train, y_train, batch_size=1, verbose = 0)
                test_err = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
                tuning_results['training error'].append(train_err)
                tuning_results['test error'].append(test_err)

    return tuning_results


## ===================================================================================================== Load Data

print("Loading Data...")
X = np.load('input_images.npy')
Y = np.load('output_images.npy')
print("Data Loaded")

# split
x_train, y_train, x_test, y_test = train_test_split(X, Y)

# normalize
x_train = max_normalize(x_train)
y_train = max_normalize(y_train)

x_test = max_normalize(x_test)
y_test = max_normalize(y_test)

# plot samples
plot_images(x_train[0:2], y_train[0:2], labels = ['input (Kn)', 'output (Ux)'])


"""
# Run Tuning
tuning_results = hpo_grid_search(x_train, y_train, x_test, y_test)

##  
best_test_err = tuning_results['test error'].index(min([abs(value) for value in tuning_results['test error']] ))
best_train_err = tuning_results['training error'].index(min([abs(value) for value in tuning_results['training error']] ))

best_setting = [tuning_results['activation'][best_test_err],
                tuning_results['loss function'][best_test_err],
                tuning_results['optimizer'][best_test_err]]

"""

best_setting = ['relu', 'mean_squared_logarithmic_error', 'adam']   # selected based on the results of grid search

## ============================================================================================ Training

best_cnn = build_cnn_model(best_setting)
model, loss, val_loss = train_model(best_cnn, x_train, y_train, epochs = 100, batch_size=5, cv_folds = 5)
model.save('model.h5', overwrite=True, save_format=None, options=None)
loss = np.array(loss)
val_loss = np.array(val_loss)
np.save('loss', loss)
np.save('val_loss', val_loss)
keras.utils.vis_utils.plot_model(model, show_shapes=True, to_file='model.png')

## ============================================================================================ Testing
from keras.models import load_model
model = load_model('model.h5')
loss = np.load('loss.npy')
val_loss = np.load('val_loss.npy')

plt.figure()
plt.plot(loss, label = 'loss')
plt.plot(val_loss, label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('mean logarithmic error')
plt.legend()

print("Training loss", model.evaluate(x_train, y_train, verbose=0))
print("Test loss", model.evaluate(x_test, y_test, verbose=0))

# predict
y_hat_train = model.predict(x_train)
y_hat_test = model.predict(x_test)

plot_images(y_test[20:22], y_hat_test[20:22], labels=['Ture', 'Predicted'])
plot_images(y_train[20:22], y_hat_train[20:22], labels=['Ture', 'Predicted'])


