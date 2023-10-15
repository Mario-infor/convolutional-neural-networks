import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

weights_history = {'w': [], 'b': []}


class MyCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        weights, _biases = model.get_weights()
        weights_history['w'].append(weights[0])
        weights_history['b'].append(_biases)


if __name__ == '__main__':
    dat_content_x = [i.strip().split() for i in open("data/ex2x.dat").readlines()]
    dat_content_y = [i.strip().split() for i in open("data/ex2y.dat").readlines()]
    for i in range(0, len(dat_content_x)):
        dat_content_x[i][0] = float(dat_content_x[i][0])
        dat_content_y[i][0] = float(dat_content_y[i][0])

    plt.scatter(dat_content_x, dat_content_y, label='Data Scatter Plot')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Practice 5')
    plt.show()

    alpha = 0.01
    training_epochs = 1000
    dat_content_x = np.asarray(dat_content_x)
    dat_content_y = np.asarray(dat_content_y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=[1]))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=alpha), loss='mean_squared_error')

    callback = MyCallback()
    train = model.fit(dat_content_x, dat_content_y, epochs=training_epochs, callbacks=[callback], verbose=False)

    print(f"First Iteration Weight: {weights_history['w'][0]}")
    print(f"First Iteration Bias: {weights_history['b'][0]}")

    print(f"Last Iteration Weight: {weights_history['w'][-1]}")
    print(f"Last Iteration Bias: {weights_history['b'][-1]}")

    plt.plot(train.history['loss'])
    plt.title('Loss Convergence')
    plt.show()

    plt.plot(weights_history['w'], label='Weight')
    plt.plot(weights_history['b'], label='Bias')
    plt.title('Weight and Bias Convergence')
    plt.legend()
    plt.show()

    prediction = model.predict(dat_content_x)

    plt.scatter(dat_content_x, dat_content_y, label='Data Scatter Plot')
    plt.plot(dat_content_x, prediction, linewidth=2, color='black', label='Lineal Regression Line')
    plt.title('Data Final Linear Approximation')
    plt.legend()
    plt.show()

    w = weights_history['w'][-1]
    b = weights_history['b'][-1]

    prediction_first_boy_height = float(b + w * 3.5)
    prediction_second_boy_height = float(b + w * 7)

    print("The 3.5 years old kid should have a height of: ", "{:.3f}".format(prediction_first_boy_height))
    print("The 7 years old kid should have a height of: ", "{:.3f}".format(prediction_second_boy_height))

    plt.scatter(dat_content_x, dat_content_y, label='Data Scatter Plot')
    plt.plot(dat_content_x, prediction, linewidth=2, color='black', label='Lineal Regression Line')
    plt.scatter([[3.5], [7]], [[prediction_first_boy_height], [prediction_second_boy_height]],
                color='red', label='Prediction for the two kids')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Prediction for the two kids')
    plt.legend()
    plt.show()
