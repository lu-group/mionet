import numpy as np
from scipy.integrate import solve_ivp

from spaces import GRF


def int_index(x, t, T):
    mat = np.linspace(0, T, x)
    return int(t / mat[1])


def ode(m, T, sensor_values1, sensor_values2):
    """ODE system"""
    s0 = [0, 0]  # initial condition

    def model(t, s):
        k = 1
        u1 = lambda t: sensor_values1[t]
        u2 = lambda t: sensor_values2[t]
        return [
            s[1] + u1(int_index(m, t, T[0])),
            -k * np.sin(s[0]) + u2(int_index(m, t, T[0])),
        ]

    res = solve_ivp(model, [0, T[0]], s0, method="RK45", t_eval=np.linspace(0, T[0], m))
    return res.y[0, :], res.y[1, :]


def run(space, m, T, num_train, num_test):
    # generate training and testing data
    print("Generating operator data...", flush=True)
    features = space.random(num_train)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values1 = space.eval_u(features, sensors)
    features = space.random(num_train)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values2 = space.eval_u(features, sensors)
    s = np.array(
        list(
            map(
                ode,
                np.tile(m, (num_train, 1)),
                np.tile(T, (num_train, 1)),
                sensor_values1,
                sensor_values2,
            )
        )
    )
    s1 = s[:, 0, :]
    s2 = s[:, 1, :]
    x = np.linspace(0, T, m)[:, None]
    X_train = [sensor_values1, sensor_values2, x]
    y_train = s1

    features = space.random(num_test)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values1 = space.eval_u(features, sensors)
    features = space.random(num_test)
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values2 = space.eval_u(features, sensors)
    s = np.array(
        list(
            map(
                ode,
                np.tile(m, (num_test, 1)),
                np.tile(T, (num_test, 1)),
                sensor_values1,
                sensor_values2,
            )
        )
    )
    s1 = s[:, 0, :]
    s2 = s[:, 1, :]
    x = np.linspace(0, T, m)[:, None]
    X_test = [sensor_values1, sensor_values2, x]
    y_test = s1

    np.savez_compressed("ODE_train.npz", X_train=X_train, y_train=y_train)
    np.savez_compressed("ODE_test.npz", X_test=X_test, y_test=y_test)


def main():
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    m = 100
    T = 1
    num_train = 1000
    num_test = 100000

    run(space, m, T, num_train, num_test)


if __name__ == "__main__":
    main()
