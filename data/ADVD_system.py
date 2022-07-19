import numpy as np

from spaces import GRF


def solve_ADVD(xmin, xmax, tmin, tmax, D, V, Nx, Nt):
    """Solve
    u_t + u_x - D * u_xx = 0
    u(x, 0) = V(x) periodic, twice continuously differentiable
    D(x) periodic, continuous
    """
    # Crank-Nicholson
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    mu = dt / h**2
    u = np.zeros([Nx, Nt])
    u[:, 0] = V
    d = D[1:]

    I = np.eye(Nx - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)

    A = (1 + d * mu) * I - (lam / 4 + d * mu / 2) * I1 + (lam / 4 - d * mu / 2) * I2
    B = 2 * I - A
    C = np.linalg.solve(A, B)

    for n in range(Nt - 1):
        u[1:, n + 1] = C @ u[1:, n]
    u[0, :] = u[-1, :]

    return x, t, u


def eval_s(m, T, Nt, sensor_values1, sensor_values2):
    return solve_ADVD(0, 1, 0, T, sensor_values1, sensor_values2, m, Nt)[2]


def run(space, m, T, Nt, num_train, num_test):
    """Advection-diffusion on the domain [0, 1] x [0, T].

    Args:
        T: Time [0, T]
        Nt: Nt in FDM
        npoints_output: For a input function, randomly choose these points from the solver output as data
    """

    print("Generating operator data...", flush=True)
    features1 = space.random(num_train)
    sensors1 = np.linspace(0, T, num=m)[:, None]
    sensor_values1 = (
        np.abs(space.eval_u(features1, np.sin(np.pi * sensors1) ** 2)) * 0.01 + 0.1
    )
    features2 = space.random(num_train)
    sensors2 = np.linspace(0, T, num=m)[:, None]
    sensor_values2 = space.eval_u(features2, np.sin(np.pi * sensors2) ** 2)
    s = np.array(
        list(
            map(
                eval_s,
                np.hstack(np.tile(m, (num_train, 1))),
                np.hstack(np.tile(T, (num_train, 1))),
                np.hstack(np.tile(Nt, (num_train, 1))),
                sensor_values1,
                sensor_values2,
            )
        )
    )
    xt = [(x, y) for x in np.linspace(0, 1, m) for y in np.linspace(0, 1, Nt)]
    s = np.reshape(s, (-1, Nt * m))
    X_train, y_train = (sensor_values1, sensor_values2, xt), s

    features1 = space.random(num_test)
    sensors1 = np.linspace(0, T, num=m)[:, None]
    sensor_values1 = (
        np.abs(space.eval_u(features1, np.sin(np.pi * sensors1) ** 2)) * 0.01 + 0.1
    )
    features2 = space.random(num_test)
    sensors2 = np.linspace(0, T, num=m)[:, None]
    sensor_values2 = space.eval_u(features2, np.sin(np.pi * sensors2) ** 2)
    s = np.array(
        list(
            map(
                eval_s,
                np.hstack(np.tile(m, (num_test, 1))),
                np.hstack(np.tile(T, (num_test, 1))),
                np.hstack(np.tile(Nt, (num_test, 1))),
                sensor_values1,
                sensor_values2,
            )
        )
    )
    xt = [(x, y) for x in np.linspace(0, 1, m) for y in np.linspace(0, 1, Nt)]
    s = np.reshape(s, (-1, Nt * m))
    X_test, y_test = (sensor_values1, sensor_values2, xt), s

    np.savez_compressed("ADVD_train.npz", X_train=X_train, y_train=y_train)
    np.savez_compressed("ADVD_test.npz", X_test=X_test, y_test=y_test)


def main():
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    m = 20
    T = 1
    Nt = 20
    num_train = 500
    num_test = 1000

    run(space, m, T, Nt, num_train, num_test)


if __name__ == "__main__":
    main()
