import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Huber

from data_transform import provide_data_for_model
from data_transform import undo_label_scaling


TEST_SET_SIZE = 0.4


"""
    This function computes the root mean squared error of the predicted
    values of a test set with the actual labels as performance measure
    for the model.
"""
def evaluate_model(predictions: np.array, y: np.array) -> float:
    n = len(predictions)
    _sum = 0
    for i in range(n):
        unsc_y = undo_label_scaling(label=float(y[i][0]))
        unsc_y_hat = undo_label_scaling(label=float(predictions[i][0]))
        _sum += (unsc_y - unsc_y_hat) ** 2
    return float((_sum / n) ** 0.5)


def train_model() -> None:
    data = provide_data_for_model()
    index = int(len(data[0]) * TEST_SET_SIZE + 0.5)
    x_train = np.array(data[0][index:])
    y_train = np.array(data[1][index:])
    x_test = np.array(data[0][:index])
    y_test = np.array(data[1][:index])

    # Building and training the model.
    tf.random.set_seed(1234)
    model = Sequential([
        tf.keras.Input(shape=(7,)),
        Dense(56, activation='relu', name = 'layer1'),
        Dense(28, activation='relu', name = 'layer2'),
        Dense(14, activation='relu', name = 'layer3'),
        Dense(7, activation='relu', name = 'layer4'),
        Dense(1, activation='linear', name = 'output_layer')
    ])
    model.compile(
        loss=Huber(), 
        optimizer=Adam(learning_rate=0.005)
    )
    model.fit(x_train, y_train, epochs=1000, batch_size=32)

    # Evaluating the model.
    predictions = model.predict(x_test)
    print(
        'Root Mean Square Error:', 
        evaluate_model(predictions=predictions, y=y_test), 
        'USD'
    )


if __name__ == '__main__':
    train_model()