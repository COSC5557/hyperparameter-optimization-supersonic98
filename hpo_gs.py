import math
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential,load_model
import keras_tuner
from tensorflow.keras.utils import plot_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt


def plot_images(imgs1, imgs2, labels=['', '']):

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


def build_cnn_model(hp, input_shape=(15, 15, 1)):
    with tf.device('/cpu:0'):
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(64, (3, 3), activation = hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D((1, 1), padding='same'))

        model.add(Conv2D(128, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))
        model.add(MaxPooling2D((3, 3), padding='same'))

        model.add(Conv2D(256, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))
        model.add(MaxPooling2D((1, 1), padding='same'))

        # Up-sampling layers
        model.add(Conv2D(256, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))
        model.add(UpSampling2D((1, 1)))

        model.add(Conv2D(128, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))
        model.add(UpSampling2D((3, 3)))

        model.add(Conv2D(64, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))
        model.add(UpSampling2D((1, 1)))

        # Output layer
        model.add(Conv2D(1, (3, 3), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh', 'softmax']), padding='same'))

        # Compile the model
        model.compile(optimizer=hp.Choice('optimizer', ['RMSprop' ,'adam', 'adamax', 'Adadelta']),
                      loss=hp.Choice('loss', ['mean_squared_error', 'mean_absolute_error',  'mean_absolute_percentage_error','mean_squared_logarithmic_error']))

    return model


def build_best_model(input_shape=(15, 15, 1)):
    with tf.device('/cpu:0'):
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D((1, 1), padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((3, 3), padding='same'))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((1, 1), padding='same'))

        # Up-sampling layers
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((1, 1)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((3, 3)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((1, 1)))

        # Output layer
        model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

        # Compile the model
        model.compile(optimizer='adamax',
                      loss='mean_squared_logarithmic_error')

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


## ===================================================================================================== Load Data
print("Loading Data...")
X = np.load('input_images.npy')
Y = np.load('output_images.npy')
print("Data Loaded")

# split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

# plot
plot_images(x_train[0:2], y_train[0:2], ['Input (Kn)', 'Output (Ux)'])

## ===================================================================================================== Preprocessing
n_comp = 225

# flatten
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1] * y_train.shape[2]))

# normalize
scaler_pre_pca_x = MinMaxScaler()
scaler_pre_pca_x.fit(x_train)
x_train = scaler_pre_pca_x.transform(x_train)
scaler_pre_pca_y = MinMaxScaler()
scaler_pre_pca_y.fit(y_train)
y_train = scaler_pre_pca_y.transform(y_train)

# reduce
pca_x = PCA(n_components=n_comp)
pca_x.fit(x_train)
x_train = pca_x.transform(x_train)
pca_y = PCA(n_components=n_comp)
pca_y.fit(y_train)
y_train = pca_y.transform(y_train)

# normalize
scaler_post_pca_x = MinMaxScaler()
scaler_post_pca_x.fit(x_train)
x_train = scaler_post_pca_x.transform(x_train)
scaler_pos_pca_y = MinMaxScaler()
scaler_pos_pca_y.fit(y_train)
y_train = scaler_pos_pca_y.transform(y_train)

# reshape
x_train = x_train.reshape((x_train.shape[0], 15, 15))
y_train = y_train.reshape((y_train.shape[0], 15, 15))

plot_images(x_train[0:2], y_train[0:2])


## ===================================================================================================== Optimization
tuner = keras_tuner.GridSearch(
    hypermodel=build_cnn_model,
    objective="val_loss",
    max_trials=64,
    executions_per_trial=1,
    overwrite=True,
    directory="gs_tuner",
    project_name="gs",
)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=42)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

print(tuner.results_summary())
best_model = tuner.get_best_models(3)
best_model = best_model[0]
print(best_model.summary())
## ===================================================================================================== Training

# train best model
x_train = np.concatenate((x_train, x_val), axis = 0)
y_train = np.concatenate((y_train, y_val), axis = 0)

# model = build_best_model(input_shape=(15, 15, 1))
plot_model(best_model, 'model.png')
model, loss, val_loss = train_model(best_model, x_train, y_train, epochs = 200, batch_size=1, cv_folds = 5)
model.save("best_model_gs.keras")
# model = load_model("best_model_trained.keras")

plt.figure()
plt.plot(loss, label = 'loss')
plt.plot(val_loss, label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('mean squared logarithmic error')
plt.legend()
plt.title("Grid Search Learning Curve")

## ===================================================================================================== model evaluation on training
y_train_err = model.evaluate(x_train, batch_size=1)
print(y_train_err)
y_train_hat = model.predict(x_train, batch_size=1)

###  inverse data
# reshape
y_train = y_train.reshape((y_train.shape[0], 15*15))
y_train_hat = y_train_hat.reshape((y_train_hat.shape[0], 15*15))

# rescale
y_tr_recon_predict = scaler_pos_pca_y.inverse_transform(y_train_hat)
y_tr_recon = scaler_pos_pca_y.inverse_transform(y_train)

# inverse PCA
y_tr_recon_predict = pca_y.inverse_transform(y_tr_recon_predict)
y_tr_recon = pca_y.inverse_transform(y_tr_recon)

# rescale
y_tr_recon_predict = scaler_pre_pca_y.inverse_transform(y_tr_recon_predict)
y_tr_recon = scaler_pre_pca_y.inverse_transform(y_tr_recon)

# reshape
y_tr_recon = y_tr_recon.reshape((y_tr_recon.shape[0], 500, 500))
y_tr_recon_predict = y_tr_recon_predict.reshape((y_tr_recon_predict.shape[0], 500, 500))

plot_images(y_tr_recon[0:2], y_tr_recon_predict[0:2], labels=['true', 'predicted'])

## ===================================================================================================== model evaluation on test
# processing test data
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
# normalize
x_test = scaler_pre_pca_x.transform(x_test)
# reduce
x_test = pca_x.transform(x_test)
# normalize
x_test = scaler_post_pca_x.transform(x_test)
# reshape
x_test = x_test.reshape((x_test.shape[0], 15, 15))
# predict
y_test_hat = model.predict(x_test, batch_size=1)

###  inverse data
# reshape
y_test_hat = y_test_hat.reshape((y_test_hat.shape[0], 15*15))
# rescale
y_tst_recon_predict = scaler_pos_pca_y.inverse_transform(y_test_hat)
# inverse PCA
y_tst_recon_predict = pca_y.inverse_transform(y_tst_recon_predict)
# rescale
y_tst_recon_predict = scaler_pre_pca_y.inverse_transform(y_tst_recon_predict)
# reshape
y_tst_recon_predict = y_tst_recon_predict.reshape((y_tst_recon_predict.shape[0], 500, 500))

plot_images(y_test[0:2], y_tst_recon_predict[0:2], labels=['true', 'predicted'])
