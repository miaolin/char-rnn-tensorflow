import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt

NUM_WORDS = 10000


def get_imdb_data():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    return train_data, train_labels, test_data, test_labels


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_index in enumerate(sequences):
        results[i, word_index] = 1.0
    return results


def build_model(layer_1_nodes, layer_2_nodes):
    model = keras.Sequential([
        keras.layers.Dense(layer_1_nodes, activation=tf.nn.relu, input_shape=(10000, )),
        keras.layers.Dense(layer_2_nodes, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.summary()
    return model


def build_model_l2(layer_1_nodes, layer_2_nodes):
    model = keras.Sequential([
        keras.layers.Dense(layer_1_nodes, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu, input_shape=(10000, )),
        keras.layers.Dense(layer_2_nodes, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.summary()
    return model


def build_model_with_dropout(layer_1_nodes, layer_2_nodes):
    dpt_model = keras.Sequential([
        keras.layers.Dense(layer_1_nodes, activation=tf.nn.relu, input_shape=(10000,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(layer_2_nodes, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    return dpt_model


def train_model(model, train_data, train_labels, test_data, test_labels):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

    history = model.fit(train_data,
                        train_labels,
                        epochs=20,
                        batch_size=512,
                        validation_data=(test_data, test_labels),
                        verbose=2)
    return history


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.show()


def main():

    train_data, train_labels, test_data, test_labels = get_imdb_data()

    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

    plt.plot(train_data[0])

    baseline_model = build_model(16, 16)
    baseline_history = train_model(baseline_model, train_data, train_labels, test_data, test_labels)

    #smaller_model = build_model(4, 4)
    #smaller_history = train_model(smaller_model, train_data, train_labels, test_data, test_labels)
    #bigger_model = build_model(512, 512)
    #bigger_history = train_model(bigger_model, train_data, train_labels, test_data, test_labels)

    #plot_history([('baseline', baseline_history),
    #              ('smaller', smaller_history),
    #              ('bigger', bigger_history)])

    model_with_l2 = build_model(16, 16)
    model_l2_history = train_model(model_with_l2, train_data, train_labels, test_data, test_labels)

    model_with_dropout = build_model_with_dropout(16, 16)
    model_dropout_history = train_model(model_with_dropout, train_data, train_labels, test_data, test_labels)

    plot_history([('baseline', baseline_history),
                  ('l2', model_l2_history),
                  ('dropout', model_dropout_history)])


if __name__ == "__main__":
    main()
