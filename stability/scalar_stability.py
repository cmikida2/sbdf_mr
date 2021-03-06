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


def adams_bashforth(y, h, lbda, order, rhs_hist):

    # These exist mostly to see where the stiffness bound is.
    if order == 1:
        y_new = y + h*(rhs_hist[0])
    elif order == 2:
        y_new = y + h*((3.0/2.0)*rhs_hist[0] - (1.0/2.0)*rhs_hist[1])
    elif order == 3:
        y_new = y + h*((23.0/12.0)*rhs_hist[0] -
                       (16.0/12.0)*rhs_hist[1] +
                       (5.0/12.0)*rhs_hist[2])
    elif order == 4:
        y_new = y + h*((55.0/24.0)*rhs_hist[0] -
                       (59.0/24.0)*rhs_hist[1] +
                       (37.0/24.0)*rhs_hist[2] -
                       (9.0/24.0)*rhs_hist[3])

    return (y_new, nonstiff(y_new, lbda),
            nonstiff(y_new, lbda) + stiff(y_new, 0))


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


def imex_bdf(y, h, t, lbda, mu, state_hist, rhs_hist, rhsns_hist,
             order, rhs_extrap):

    # Now we can perform the macrostep stuff, basically as normal.
    nonstiff_cont = nonstiff_extrap(y, h, t + h, lbda, mu,
                                    rhsns_hist, state_hist,
                                    order, rhs_extrap)

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
    return (y_new, new_nonstiff,
            new_nonstiff + stiff(y_new, mu))
    # no correction on nonstiff term?
    # return (y_new, nonstiff_cont/h, nonstiff_cont/h +
    #         stiff(y_new, mu))


def main():

    # Stability test
    t_start = 0
    t_end = 100
    dt = 1
    order = 4
    # Modify this to see what effect stiffness has on
    # the stability of the explicit part of the SBDF.
    mu = 0

    # Set ratio criteria - order 4 is persnickety
    if order < 4:
        ratio_threshold = 9
    else:
        ratio_threshold = 3

    # root locus: loop through theta.
    thetas = np.linspace(-np.pi, np.pi, 500)
    # thetas = np.linspace(-np.pi, np.pi, 50)
    rs = []
    crits = []

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    for theta in thetas:

        # Find stability boundary iteratively,
        # for now via simple bisection.
        r = 2
        r_good = 0
        r_bad = 100
        dr = 100
        while abs(dr) > 0.000001:

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
            state_hist = np.empty((order), dtype=complex)
            rhs_hist[0] = nonstiff(y_old, lbda) + stiff(y_old, mu)
            rhsns_hist[0] = nonstiff(y_old, lbda)
            state_hist[0] = y_old
            tiny = 1e-15
            fail = False
            ratio = 0
            ratio_counter = 0
            while t < t_end - tiny:
                if step < order - 1:
                    # "Bootstrap" using known exact solution.
                    y = exact(t + dt, mu, lbda)
                    dy_ns = nonstiff(y, lbda)
                    dy_full = dy_ns + stiff(y, mu)
                else:
                    y, dy_ns, dy_full = imex_bdf(y_old, dt, t, lbda, mu,
                                                 state_hist,
                                                 rhs_hist,
                                                 rhsns_hist, order,
                                                 False)
                # Rotate histories.
                for i in range(order-1, 0, -1):
                    rhs_hist[i] = rhs_hist[i-1]
                    rhsns_hist[i] = rhsns_hist[i-1]
                    state_hist[i] = state_hist[i-1]
                rhs_hist[0] = dy_full
                rhsns_hist[0] = dy_ns
                state_hist[0] = y
                # Append to states and prepare for next step.
                states.append(y)
                exact_states.append(exact(t + dt, mu, lbda))
                t += dt
                times.append(t)
                y_old = y
                ratio = abs(states[-1]/states[-2])
                if ratio > 1.0:
                    ratio_counter += 1
                else:
                    ratio_counter = 0
                if ratio_counter > ratio_threshold:
                    fail = True
                    break
                step += 1
            if fail:
                # Failed - decrease r.
                dr = r - (r_bad + r_good)/2
                cand = (r_bad + r_good)/2
                r_bad = r
                r = cand
                # print("Radius failed")
            else:
                # Success: increase r.
                dr = r - (r_bad + r_good)/2
                cand = (r_bad + r_good)/2
                r_good = r
                r = cand

        print("Theta = ", theta)
        print("r = ", r_good)
        rs.append(r_good)

        # Evaluate our new expression to check if we are on the right
        # track with this...
        lbda = r_good*np.exp(theta*1j)
        if order == 1:
            crit = abs((1 + lbda)/(1 - mu))
            print("Max Stability Criterion Value: ", crit)
            crits.append(crit)
        elif order == 2:
            a = 3/2 - mu
            b = -(2*lbda + 2)
            c = lbda + 1/2
            crit1 = abs((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
            crit2 = abs((-b - np.sqrt(b**2 - 4*a*c)) / (2*a))
            print("Max Stability Criterion Value: ", max(crit1, crit2))
            crits.append(max(crit1, crit2))
        elif order == 3:
            # Order 3 has a very complicated polynomial
            crit = max(abs(np.roots([11/6 - mu, -3*(lbda + 1),
                                     3*(lbda + 1/2), -(lbda + 1/3)])))
            print("Max Stability Criterion Value: ", crit)
            crits.append(crit)
        else:
            # Order 4 has a very complicated polynomial
            crit = max(abs(np.roots([1 - (12/25)*mu, -(48/25)*(lbda + 1),
                                     (36/25)*(2*lbda + 1),
                                     -(16/25)*(3*lbda + 1),
                                     (3/25)*(4*lbda + 1)])))
            print("Max Stability Criterion Value: ", crit)
            crits.append(crit)

    reals = np.real(rs*np.exp(1j*thetas))
    imags = np.imag(rs*np.exp(1j*thetas))
    # Save this data to a file.
    # np.savetxt("explicit/reals_{}.txt".format(order), reals)
    # np.savetxt("explicit/imags_{}.txt".format(order), imags)

    # np.savetxt("explicit/rs_{}.txt".format(order), rs)
    # np.savetxt("explicit/thetas_{}.txt".format(order), thetas)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(reals, imags, 'r--')
    plt.title("Stability: Order-{order} SBDF w/Scalar Test, mu = {mu}".format(
        order=order, mu=mu))
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.show()

    plt.clf()
    plt.plot(thetas, crits, 'r--')
    plt.title("Stability Criterion: Order-{order} mu = {mu}".format(
        order=order, mu=mu))
    plt.xlabel("theta")
    plt.ylabel("max(z)")
    plt.show()


if __name__ == "__main__":
    main()
