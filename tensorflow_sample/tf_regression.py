import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    boston_housing = keras.datasets.boston_housing
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

    # shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print("Training set: {}".format(train_data.shape))
    print("Testing set: {}".format(test_data.shape))
    return train_data, train_labels, test_data, test_labels


def normalize_features(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    return train_data, test_data


def build_model(num_of_feature):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(num_of_feature, )),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val Loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


def main():

    EPOCHS = 500
    train_data, train_labels, test_data, test_labels = load_data()
    print(train_data[0])

    train_data, test_data = normalize_features(train_data, test_data)
    num_of_features = train_data.shape[1]

    model = build_model(num_of_features)
    model.summary()

    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])
    plot_history(history)

    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

    test_predictions = model.predict(test_data).flatten()
    print(test_predictions)


if __name__ == "__main__":
    main()
