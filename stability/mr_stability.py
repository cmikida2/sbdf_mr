import numpy as np


def nonstiff(y, lbda):

    dy = np.empty(1, dtype=complex)
    dy = lbda*y
    return dy


def stiff(y, mu):

    dy = np.empty(1, dtype=complex)
    dy = mu*y
    return dy


def exact(t, mu, lbda):
    y = np.empty(1, dtype=complex)
    y = np.exp((lbda+mu)*t)
    return y


def adams_bashforth(y, h, order, time_hist, rhs_hist, t):

    if order == 1:
        y_new = y + h*(rhs_hist[0])
        return y_new

    # Now that we have disparate time history points, we need to
    # solve for the AB coefficients.
    thist = time_hist.copy() - t
    ab = np.zeros(order)
    vdmt = np.zeros((order, order))
    coeff_rhs = np.zeros(order)
    for i in range(0, order):
        coeff_rhs[i] = (1/(i+1))*h**(i+1)
        for j in range(0, order):
            vdmt[i][j] = thist[i]**j

    ab = np.linalg.solve(np.transpose(vdmt), coeff_rhs)

    # Obtain new substep-level state estimate with these coeffs.
    if order == 2:
        y_new = y + ab[0]*rhs_hist[0] + ab[1]*rhs_hist[1]
    elif order == 3:
        y_new = y + (ab[0]*rhs_hist[0] +
                     ab[1]*rhs_hist[1] +
                     ab[2]*rhs_hist[2])
    elif order == 4:
        y_new = y + (ab[0]*rhs_hist[0] +
                     ab[1]*rhs_hist[1] +
                     ab[2]*rhs_hist[2] +
                     ab[3]*rhs_hist[3])

    return y_new


def nonstiff_extrap(y, h, t, lbda, mu, rhsns_hist, state_hist, order,
                    rhs_extrap):

    # Return nonstiff RHS contribution to be used in implicit solve!

    if rhs_extrap is False:
        # Option 1: use "prediction" value in BDF.
        if order == 1:
            y_new = state_hist[0]
        elif order == 2:
            y_new = 2*state_hist[0] - state_hist[1]
        elif order == 3:
            y_new = 3*state_hist[0] - 3*state_hist[1] + state_hist[2]
        else:
            y_new = 4*state_hist[0] - 6*state_hist[1] + \
                    4*state_hist[2] - state_hist[3]

        return h*nonstiff(y_new, lbda)
    else:
        # Option 2: use RHS histories instead, a la AB.
        # (Since this is only done to obtain the final nonstiff
        # RHS extrapolation, we know we will have a constant
        # step size)
        if order == 1:
            a_new = rhsns_hist[0]
        elif order == 2:
            a_new = 2*rhsns_hist[0] - rhsns_hist[1]
        elif order == 3:
            a_new = 3*rhsns_hist[0] - 3*rhsns_hist[1] + rhsns_hist[2]
        else:
            a_new = 4*rhsns_hist[0] - 6*rhsns_hist[1] + \
                    4*rhsns_hist[2] - rhsns_hist[3]

        return h*a_new


def stiff_interp(h, rhs_hist, order):

    if order == 1:
        rhs_new = rhs_hist[1] + (rhs_hist[0] - rhs_hist[1])*h
    elif order == 2:
        rhs_new = (rhs_hist[0]*h*((h+1)/2) + rhs_hist[1]*((h-1)/-1)*(h+1)
                   + rhs_hist[2]*(h/-1)*((h-1)/-2))
    elif order == 3:
        rhs_new = (rhs_hist[0]*h*((h+1)/2)*((h+2)/3) +
                   rhs_hist[1]*((h-1)/-1)*(h+1)*((h+2)/2) +
                   rhs_hist[2]*(h/-1)*((h-1)/-2)*(h+2) +
                   rhs_hist[3]*((h+1)/-1)*(h/-2)*((h-1)/-3))
    else:
        rhs_new = (rhs_hist[0]*h*((h+1)/2)*((h+2)/3)*((h+3)/4) +
                   rhs_hist[1]*((h-1)/-1)*(h+1)*((h+2)/2)*((h+3)/3) +
                   rhs_hist[2]*(h/-1)*((h-1)/-2)*(h+2)*((h+3)/2) +
                   rhs_hist[3]*((h+1)/-1)*(h/-2)*((h-1)/-3)*(h+3) +
                   rhs_hist[4]*((h+1)/-2)*(h/-3)*((h-1)/-4)*((h+2)/-1))

    return rhs_new


def imex_bdf(y, h, t, lbda, mu, state_hist, rhs_hist, rhsns_hist,
             order, sr):

    # "Slowest first:" - first, use RHS extrapolation to form implicit
    # prediction, then solve.
    nonstiff_cont = nonstiff_extrap(y, h, t + h, lbda, mu,
                                    rhsns_hist, state_hist,
                                    order, True)

    # Solve.
    if order == 1:
        y_new = (nonstiff_cont + state_hist[0])/(1 - h*mu)
    elif order == 2:
        y_new = ((2.0/3.0)*nonstiff_cont + (4.0/3.0)*state_hist[0]
                 - (1.0/3.0)*state_hist[1])/(1 - (2.0/3.0)*h*mu)
    elif order == 3:
        y_new = ((6.0/11.0)*nonstiff_cont + (18.0/11.0)*state_hist[0]
                 - (9.0/11.0)*state_hist[1] +
                 (2.0/11.0)*state_hist[2])/(1 - (6.0/11.0)*h*mu)
    else:
        y_new = ((12.0/25.0)*nonstiff_cont + (48.0/25.0)*state_hist[0]
                 - (36.0/25.0)*state_hist[1] +
                 (16.0/25.0)*state_hist[2] -
                 (3.0/25.0)*state_hist[3])/(1 - (12.0/25.0)*h*mu)

    new_nonstiff = nonstiff(y_new, lbda)

    # Now do AB at the substep level, supplementing with interpolation of the
    # newfound stiff RHS.
    new_stiff = stiff(y_new, mu)

    if sr > 1:
        # For the first substep, use existing RHS history...
        substep_rhs_hist = rhsns_hist.copy() + rhs_hist.copy()
        substep_time_hist = np.zeros(order)
        for i in range(0, order):
            substep_time_hist[i] = t - i*h

        y_substep = adams_bashforth(y, (1/sr)*h, order,
                                    substep_time_hist, substep_rhs_hist,
                                    substep_time_hist[0])
        # Now build a stiff RHS history so we can interpolate with it.
        new_rhs_hist = np.empty(order+1, dtype=complex)
        for i in range(0, order):
            new_rhs_hist[i+1] = rhs_hist[i]
        new_rhs_hist[0] = new_stiff

        for j in range(1, sr):
            # Interpolate to obtain stiff RHS.
            stiff_rhs = stiff_interp((j/sr), new_rhs_hist, order)
            # Evaluate to obtain nonstiff RHS.
            nonstiff_rhs = nonstiff(y_substep, lbda)
            # Combine and rotate substep-level history and time history.
            # Rotate this into history.
            for i in range(order-1, 0, -1):
                substep_rhs_hist[i] = substep_rhs_hist[i-1]
                substep_time_hist[i] = substep_time_hist[i-1]
            substep_rhs_hist[0] = nonstiff_rhs + stiff_rhs
            substep_time_hist[0] = t + (j/sr)*h
            y_substep = adams_bashforth(y_substep, (1/sr)*h, order,
                                        substep_time_hist, substep_rhs_hist,
                                        substep_time_hist[0])

        new_nonstiff = nonstiff(y_substep, lbda)
        # FIXME: re-solve?
        if order == 1:
            y_new = (new_nonstiff*h + state_hist[0])/(1 - h*mu)
        elif order == 2:
            y_new = ((2.0/3.0)*new_nonstiff*h + (4.0/3.0)*state_hist[0]
                     - (1.0/3.0)*state_hist[1])/(1 - (2.0/3.0)*h*mu)
        elif order == 3:
            y_new = ((6.0/11.0)*new_nonstiff*h + (18.0/11.0)*state_hist[0]
                     - (9.0/11.0)*state_hist[1] +
                     (2.0/11.0)*state_hist[2])/(1 - (6.0/11.0)*h*mu)
        else:
            y_new = ((12.0/25.0)*new_nonstiff*h + (48.0/25.0)*state_hist[0]
                     - (36.0/25.0)*state_hist[1] +
                     (16.0/25.0)*state_hist[2] -
                     (3.0/25.0)*state_hist[3])/(1 - (12.0/25.0)*h*mu)

    new_stiff = stiff(y_new, mu)
    # FIXME: correction on nonstiff term?
    new_nonstiff = nonstiff(y_new, lbda)
    return (y_new, new_nonstiff, new_stiff)


def main():

    # Stability test
    t_start = 0
    t_end = 100
    dt = 1
    order = 1
    srs = [1, 2, 3, 4, 5]

    n_thetas = 100
    # root locus: loop through theta.
    thetas = np.linspace(-np.pi, np.pi, n_thetas)
    theta_mu = -np.pi
    # rs_mu = [0.01, 0.1, 1, 10, 100]
    rs_mu = [0]

    # Set up plotting
    reals_std = np.loadtxt("reals_{}.txt".format(order))
    imags_std = np.loadtxt("imags_{}.txt".format(order))
    import matplotlib.pyplot as plt
    plt.clf()

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    for sr in srs:
        rs_max = []
        for _i in range(0, n_thetas):
            rs_max.append(1000)
        for r_mu in rs_mu:
            mu = r_mu*np.exp(theta_mu*1j)
            # We will be finding the minimum (limiting)
            # lambdas for all mus in the left half plane.
            for ith, theta in enumerate(thetas):

                # Find stability boundary iteratively,
                # for now via simple bisection.
                r = 2
                r_good = 0
                r_bad = 110
                dr = 110
                while abs(dr) > 0.00001:

                    lbda = r*np.exp(theta*1j)
                    y_old = 1

                    times = []
                    states = []
                    states.append(y_old)
                    exact_states = []
                    exact_states.append(y_old)

                    t = t_start
                    times.append(t)
                    step = 0
                    rhs_hist = np.empty((order), dtype=complex)
                    rhsns_hist = np.empty((order), dtype=complex)
                    state_hist = np.empty((order+1), dtype=complex)
                    rhs_hist[0] = (stiff(y_old, mu) + nonstiff(y_old, lbda))
                    rhsns_hist[0] = nonstiff(y_old, lbda)
                    state_hist[0] = y_old
                    tiny = 1e-15
                    while t < t_end - tiny:
                        if step < order:
                            # "Bootstrap" using known exact solution.
                            y = exact(t + dt, mu, lbda)
                            dy_ns = nonstiff(y, lbda)
                            dy_full = dy_ns + stiff(y, mu)
                        else:
                            y, dy_ns, dy_full = imex_bdf(y_old, dt, t, lbda,
                                                         mu, state_hist,
                                                         rhs_hist,
                                                         rhsns_hist, order,
                                                         sr)
                        # Rotate histories.
                        for i in range(order-1, 0, -1):
                            rhs_hist[i] = rhs_hist[i-1]
                            rhsns_hist[i] = rhsns_hist[i-1]
                        for i in range(order, 0, -1):
                            state_hist[i] = state_hist[i-1]
                        rhs_hist[0] = dy_full
                        rhsns_hist[0] = dy_ns
                        state_hist[0] = y
                        # Append to states and prepare for next step.
                        states.append(y)
                        t += dt
                        times.append(t)
                        y_old = y
                        step += 1
                    if abs(states[-1]/states[-2]) > 1.0 or np.isnan(
                            abs(states[-1])):
                        # Failed - decrease r.
                        dr = r - (r_bad + r_good)/2
                        cand = (r_bad + r_good)/2
                        r_bad = r
                        r = cand
                    else:
                        # Success: increase r.
                        dr = r - (r_bad + r_good)/2
                        cand = (r_bad + r_good)/2
                        r_good = r
                        r = cand

                if r_good < rs_max[ith]:
                    rs_max[ith] = r_good

        reals = np.real(rs_max*np.exp(1j*thetas))
        imags = np.imag(rs_max*np.exp(1j*thetas))
        plt.plot(reals, imags, label="SR={}".format(sr))

    plt.plot(reals_std, imags_std, 'g--', label="Single-Rate Region")
    plt.title("Explicit Stability Search, Order {order}".format(
        order=order))
    plt.legend()
    plt.show()
    # plt.savefig("second_order_rflow.png")


if __name__ == "__main__":
    main()
