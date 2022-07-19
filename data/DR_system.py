import numpy as np

from spaces import GRF


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def eval_s(m, k, T, Nt, sensor_values1, sensor_values2):
    return solve_ADR(
        0,
        1,
        0,
        T,
        lambda x: 0.01 * (1 + abs(sensor_values1)),
        lambda x: np.zeros_like(x),
        lambda u: k * u**2,
        lambda u: 2 * k * u,
        lambda x, t: np.tile(sensor_values2[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        m,
        Nt,
    )[2]


def run(space, m, k, T, Nt, num_train, num_test):
    """Diffusion-reaction on the domain [0, 1] x [0, T].

    Args:
        T: Time [0, T]
        Nt: Nt in FDM
        npoints_output: For a input function, randomly choose these points from the solver output as data
    """
    print("Generating operator data...", flush=True)
    xmin = 0
    xmax = 1
    tmin = 0
    tmax = T
    npoints_output = Nt * m

    features = space.random(num_train)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values1 = space.eval_u(features, sensors)
    features = space.random(num_train)
    sensor_values2 = space.eval_u(features, sensors)
    s = list(
        map(
            eval_s,
            np.hstack(np.tile(m, (num_train, 1))),
            np.hstack(np.tile(k, (num_train, 1))),
            np.hstack(np.tile(T, (num_train, 1))),
            np.hstack(np.tile(Nt, (num_train, 1))),
            sensor_values1,
            sensor_values2,
        )
    )
    xt = [(x, y) for x in np.linspace(0, 1, m) for y in np.linspace(0, T, Nt)]
    s = np.reshape(s, (-1, npoints_output))
    X_train, y_train = (sensor_values1, sensor_values2, xt), s

    sensors = np.linspace(0, 1, num=m)[:, None]
    features = space.random(num_test)
    sensor_values1 = space.eval_u(features, sensors)
    features = space.random(num_test)
    sensor_values2 = space.eval_u(features, sensors)
    s = list(
        map(
            eval_s,
            np.hstack(np.tile(m, (num_test, 1))),
            np.hstack(np.tile(k, (num_test, 1))),
            np.hstack(np.tile(T, (num_test, 1))),
            np.hstack(np.tile(Nt, (num_test, 1))),
            sensor_values1,
            sensor_values2,
        )
    )
    s = np.reshape(s, (-1, npoints_output))
    xt = [(x, y) for x in np.linspace(0, 1, m) for y in np.linspace(0, T, Nt)]
    X_test, y_test = (sensor_values1, sensor_values2, xt), s

    np.savez_compressed("DR_train.npz", X_train=X_train, y_train=y_train)
    np.savez_compressed("DR_test.npz", X_test=X_test, y_test=y_test)


def main():
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    m = 100
    k = 0.01
    T = 1
    Nt = 100
    num_train = 1000
    num_test = 5000

    run(space, m, k, T, Nt, num_train, num_test)


if __name__ == "__main__":
    main()
