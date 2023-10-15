import matplotlib.pyplot as plt
import numpy as np

cost_list = []


def relu(x):
    return np.maximum(0, x)


def create_mini_batches(mb_size, x, y, shuffle=True):
    total_data = x.shape[0]
    if shuffle:
        pos = np.arange(total_data)
        np.random.shuffle(pos)
        x = x[pos]
        y = y[pos]

    return ((x[i:i + mb_size], y[i:i + mb_size]) for i in range(0, total_data, mb_size))


def init_parameters(input_size, neurons):
    w1 = np.random.rand(neurons[0], input_size) * 0.001
    b1 = np.zeros((neurons[0], 1))

    w2 = np.random.rand(neurons[1], neurons[0]) * 0.001
    b2 = np.zeros((neurons[1], 1))

    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def scores(x, parameters):
    z1 = parameters['w1'] @ x + parameters['b1']
    a1 = relu(z1)
    z2 = parameters['w2'] @ a1 + parameters['b2']

    return z2, z1, a1


def softmax(x):
    exp_scores = np.exp(x)
    sum_exp_scores = np.sum(exp_scores, axis=0)
    probs = exp_scores / sum_exp_scores

    return probs


def x_entropy(scores, y, batch_size):
    probs = softmax(scores)
    y_hat = probs[y.squeeze(), np.arange(batch_size)]
    cost = np.sum(-np.log(y_hat)) / batch_size

    return probs, cost


def backward(probs, x, y, z1, a1, parameters, batch_size):
    grads = {}
    probs[y.squeeze(), np.arange(batch_size)] -= 1
    dz2 = probs.copy()

    dw2 = dz2 @ a1.T / batch_size
    db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
    da1 = parameters['w2'].T @ dz2

    dz1 = da1.copy()
    dz1[z1 <= 0] = 0

    dw1 = dz1 @ x
    db1 = np.sum(dz1, axis=1, keepdims=True)

    grads = {'w1': dw1, 'b1': db1, 'w2': dw2, 'b2': db2}

    return grads


def accuracy(x_data, y_data, mb_size, parameters):
    correct = 0
    total = 0
    for i, (x, y) in enumerate(create_mini_batches(mb_size, x_data, y_data)):
        scores_2, z1, a1 = scores(x.T, parameters)
        y_hat, cost = x_entropy(scores_2, y, batch_size=len(x))

        correct += np.sum(np.argmax(y_hat, axis=0) == y.squeeze())
        total += y_hat.shape[1]

    return correct / total


def train(epochs, parameters, mb_size, learning_rate):
    cost = -1
    for epoch in range(epochs):
        for i, (x, y) in enumerate(create_mini_batches(mb_size, x_train, y_train)):
            scores_2, z1, a1 = scores(x.T, parameters)
            y_hat, cost = x_entropy(scores_2, y, batch_size=len(x))
            grads = backward(y_hat, x, y, z1, a1, parameters, batch_size=len(x))

            parameters['w1'] = parameters['w1'] - learning_rate * grads['w1']
            parameters['b1'] = parameters['b1'] - learning_rate * grads['b1']
            parameters['b2'] = parameters['b2'] - learning_rate * grads['b2']
            parameters['w2'] = parameters['w2'] - learning_rate * grads['w2']

        print(f'Costo es: {cost}, y accuracy: {accuracy(x_test, y_test, mb_size, parameters)}')
        # Este cost_list se usó para poder hacre la gráfica de los costos
        # cost_list.append(cost)

    return parameters


def predict(x):
    scores_predict, _, _ = scores(x, parameters)
    return np.argmax(scores_predict)


if __name__ == '__main__':
    data = [i.strip().split() for i in open("data/mnist.txt").readlines()]
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            data[i][j] = int(data[i][j])
            if j != (len(data[i]) - 1):
                data[i][j] = data[i][j] / 255

    x_train = data[:900]
    y_train = []
    for i in range(0, len(x_train)):
        y_train.append(x_train[i].pop())

    x_test = data[900:1000]
    y_test = []
    for i in range(0, len(x_test)):
        y_test.append(x_test[i].pop())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    mb_size = 1

    # These values are used when training the neural network
    # parameters = init_parameters(28 * 28, [200, 10])
    # learning_rate = 1e-2
    # epochs = 20

    # This is the method that starts the training process
    # parameters = train(epochs, parameters, mb_size, learning_rate)

    # This is the part that saves the parameters in files after being trained
    # np.savetxt("data/w1.txt", parameters['w1'])
    # np.savetxt("data/b1.txt", parameters['b1'])
    # np.savetxt("data/w2.txt", parameters['w2'])
    # np.savetxt("data/b2.txt", parameters['b2'])

    # This is the part where the previously trained and saved parameters are read
    w1 = np.loadtxt('data/w1.txt')
    b1 = (np.loadtxt('data/b1.txt')).reshape(-1, 1)
    w2 = np.loadtxt('data/w2.txt')
    b2 = (np.loadtxt('data/b2.txt')).reshape(-1, 1)
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # This is the part that was used to graph the costs in each epoch, it is necessary to do the
    # training process to be able to make this graph since the list of costs is not being stored
    # x = np.arange(1, 21)
    # y = cost_list
    # plt.plot(x, y)
    # plt.xlabel('Epoch number')
    # plt.ylabel('Cost Value')
    # plt.title('Practice 4')
    # plt.show()

    # This is the part where the network's percentage of successes for the training set is calculated
    # print("Accuracy Train: ", accuracy(x_train, y_train, mb_size, parameters))

    # This is the part where the network's success rate for the test set is calculated
    accuracy_result = accuracy(x_test, y_test, mb_size, parameters)
    # print("Accuracy Test: ", accuracy_result)

    # This is a pie chart to observe the percentage of successes of the network with the test set
    values = [accuracy_result, 1 - accuracy_result]
    names = [f"Correct ({accuracy_result * 100} %)", f"Incorrect ({(1 - accuracy_result) * 100} %)"]
    plt.pie(values, labels=names)
    plt.title('Recognition percentage of the NN with the test set')
    plt.show()

    # This is the part that was used to show the examples where the neural network made a mistake
    # in classifying the images
    # idx = np.random.randint(len(x_test)-1)
    # print("The chosen value is: " + str(y_test[idx][-1]))
    # prediction = predict(x_test[idx].reshape(-1, 1))
    # print(f'The predicted value is: {prediction}')
