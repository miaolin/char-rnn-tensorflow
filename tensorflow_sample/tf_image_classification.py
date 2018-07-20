import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def prepare_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return train_images, train_labels, test_images, test_labels


def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.gca().grid(False)
    plt.show()


def preprocessing(train_images, test_images):
    return train_images / 255.0, test_images / 255.0


def plot_sample_images(train_images, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


def plot_images_prediction(test_images, test_labels, predictions):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'
        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    class_names[true_label]),
                   color=color)
    plt.show()


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, number_of_epochs):
    model.fit(train_images, train_labels, epochs=number_of_epochs)
    return model


def main():

    train_images, train_labels, test_images, test_labels = prepare_data()
    print(np.unique(train_labels, return_counts=True))
    print(np.unique(test_labels, return_counts=True))

    print(train_images.shape)
    print(test_images.shape)

    train_images, test_images = preprocessing(train_images, test_images)
    model = build_model()
    model = train_model(model, train_images, train_labels, 5)

    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print(predictions[0])

    plot_images_prediction(test_images, test_labels, predictions)


if __name__ == "__main__":
    main()