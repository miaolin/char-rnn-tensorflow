import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

vocab_size = 10000


def get_imdb_data(imdb):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    return train_data, train_labels, test_data, test_labels


def generate_word_index(imdb):
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    for n in range(10):
        print(reverse_word_index[n])
    return word_index, reverse_word_index


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def preprocessing(data, word_index):
    data = keras.preprocessing.sequence.pad_sequences(data,
                                                      value=word_index["<PAD>"],
                                                      padding='post',
                                                      maxlen=256)
    return data


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def separate_data(train_data, train_labels, validate_nums=10000):
    x_val = train_data[:validate_nums]
    partial_x_train = train_data[validate_nums:]

    y_val = train_labels[:validate_nums]
    partial_y_train = train_labels[validate_nums:]
    return x_val, partial_x_train, y_val, partial_y_train


def model_train(partial_x_train, partial_y_train, x_val, y_val, model):
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)
    return model, history


def plot_training_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.plot(epochs, acc, 'r*', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


def main():

    imdb = keras.datasets.imdb
    train_data, train_labels, test_data, test_labels = get_imdb_data(imdb)
    print(len(train_data[0]), len(train_data[1]))

    word_index, reverse_word_index = generate_word_index(imdb)
    print(decode_review(train_data[0], reverse_word_index))

    train_data = preprocessing(train_data, word_index)
    test_data = preprocessing(test_data, word_index)

    x_val, partial_x_train, y_val, partial_y_train = separate_data(train_data, train_labels)
    model = build_model()
    model, history = model_train(partial_x_train, partial_y_train, x_val, y_val, model)

    plot_training_results(history)


if __name__ == "__main__":
    main()
