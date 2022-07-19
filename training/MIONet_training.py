import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
from deepxde.nn.tensorflow_compat_v1.mionet import MIONetCartesianProd
from deepxde.data.quadruple import QuadrupleCartesianProd


def network(problem, m):
    if problem == "ODE":
        branch = [m, 200, 200]
        trunk = [1, 200, 200]
    elif problem == "DR":
        branch = [m, 200, 200]
        trunk = [2, 200, 200]
    elif problem == "ADVD":
        branch = [m, 300, 300, 300]
        trunk = [2, 300, 300, 300]
    return branch, trunk


def run(problem, lr, epochs, m, activation, initializer):
    training_data = np.load("../data/" + problem + "_train.npz", allow_pickle=True)
    testing_data = np.load("../data/" + problem + "_test.npz", allow_pickle=True)
    X_train = training_data["X_train"]
    y_train = training_data["y_train"]
    X_test = testing_data["X_test"]
    y_test = testing_data["y_test"]

    branch_net, trunk_net = network(problem, m)

    data = QuadrupleCartesianProd(X_train, y_train, X_test, y_test)
    net = MIONetCartesianProd(
        branch_net,
        branch_net,
        trunk_net,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        initializer,
        regularization=None,
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    checker = dde.callbacks.ModelCheckpoint(
        "model/mionet_model.ckpt", save_better_only=True, period=1000
    )
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    print(
        "# Parameters:",
        np.sum(
            [
                np.prod(v.get_shape().as_list())
                for v in tf.compat.v1.trainable_variables()
            ]
        ),
    )


def main():
    # Problems:
    # - "ODE": Antiderivative, Nonlinear ODE
    # - "DR": Diffusion-reaction
    # - "ADVD": Advection-diffusion
    problem = "ODE"
    T = 1
    m = 100
    lr = 0.0002 if problem in ["ADVD"] else 0.001
    epochs = 100000
    activation = (
        ["relu", None, "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    )
    initializer = "Glorot normal"

    run(problem, lr, epochs, m, activation, initializer)


if __name__ == "__main__":
    main()
