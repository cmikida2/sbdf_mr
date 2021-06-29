import numpy as np
import matplotlib.pyplot as plt


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


def lagrange(t, hist, time_hist, order):

    # We use this routine to interpolate or extrapolate, given a history,
    # an accompanying time history, and a new time at which to obtain a
    # value.
    a_new = 0
    for i in range(0, order):
        term = hist[i]
        for j in range(0, order):
            if i != j:
                term = term*(t - time_hist[j])/(time_hist[i] - time_hist[j])
        a_new += term

    return a_new


def nonstiff_extrap(y, h, lbda, mu, rhsns_hist, state_hist, order,
                    rhs_extrap):

    # Return nonstiff RHS contribution to be used in implicit solve
    if rhs_extrap is False:
        # Option 1: use "prediction" value in BDF (state extrap)
        # (for fractional steps, these are derived via Lagrange
        # extrapolation)
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


def imex_mr_bdf(y, h, t, lbda, mu, state_hist, rhs_hist,
                rhsns_hist, time_hist, order, sr):

    # "Slowest first:" - first, use RHS extrapolation to form implicit
    # prediction, then solve.
    nonstiff_cont = h*lagrange(t + h, rhsns_hist, time_hist, order)

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
        new_time_hist = np.zeros(order+1)
        for i in range(0, order):
            new_rhs_hist[i+1] = rhs_hist[i]
            new_time_hist[i+1] = time_hist[i]
        new_rhs_hist[0] = new_stiff
        new_time_hist[0] = t + h

        for j in range(1, sr):
            # Interpolate to obtain stiff RHS.
            stiff_rhs = lagrange(t + (j/sr)*h, new_rhs_hist, new_time_hist,
                                 order + 1)
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

        # Re-solve.
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

    # print("AB Coeffs:")
    # print(ab/h)

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


def plot_individual_run(dts, errors, sr):
    z1 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[0, :]))), 1)
    z2 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[1, :]))), 1)
    z3 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[2, :]))), 1)
    z4 = np.polyfit(np.log10(np.array(dts)),
                    np.log10(np.abs(np.array(errors[3, :]))), 1)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    p3 = np.poly1d(z3)
    p4 = np.poly1d(z4)

    plt.clf()
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[0, :]))), c="g")
    plt.plot(np.log10(np.array(dts)), p1(np.log10(np.array(dts))), "g--")
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[1, :]))), c="b")
    plt.plot(np.log10(np.array(dts)), p2(np.log10(np.array(dts))), "b--")
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[2, :]))), c="r")
    plt.plot(np.log10(np.array(dts)), p3(np.log10(np.array(dts))), "r--")
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[3, :]))), c="k")
    plt.plot(np.log10(np.array(dts)), p4(np.log10(np.array(dts))), "k--")
    plt.legend(["EOC=%.6f" % (z1[0]), "EOC=%.6f" % (z2[0]),
                "EOC=%.6f" % (z3[0]), "EOC=%.6f" % (z4[0]),
                "Order 1 Data, State", "Order 2 Data, State",
                "Order 3 Data, State", "Order 4 Data, State"])
    plt.xlabel("log10(dt)")
    plt.ylabel("log10(err)")
    plt.ylim((-16, -1))
    plt.title("Log-Log Error Timestep Plots: MR SBDF, SR = %d" % sr)
    plt.show()


def main():

    # some new IMEX methods.
    t_start = 0
    # t_end = 1
    t_end = 0.5
    # t_end = 0.05
    dts = [0.05, 0.01, 0.005, 0.001]
    # orders = [1, 2, 3, 4]
    orders = [2]
    errors = np.zeros((4, 4))
    srs = [1, 2, 3, 4, 5]
    # srs = [1, 2]
    est_orders = np.zeros((4, 5))
    # mu = 0  # explicit example
    # lbda = -1  # explicit example
    mu = -50  # stiff-controlled example
    lbda = 0.5  # stiff-controlled example
    # Kaps linearization eigenvalues:
    # -1.00399812e+03 -1.00199410e+00
    # mu = -1.00399812e+03 * 0.05  # experiment
    # lbda = -1.00199410 * 0.05  # experiment
    plot_runs = False

    for sr_i, sr in enumerate(srs):
        for j, order in enumerate(orders):
            from pytools.convergence import EOCRecorder
            eocrec = EOCRecorder()
            for k, dt in enumerate(dts):

                if order == 1:
                    crit = abs((1 + lbda*dt)/(1 - mu*dt))
                    print("Criteria: ", crit)
                elif order == 2:
                    a = 3/2 - mu*dt
                    b = -(2*lbda*dt + 2)
                    c = lbda*dt + 1/2
                    crit1 = abs((-b + np.lib.scimath.sqrt(b**2 - 4*a*c))
                                / (2*a))
                    crit2 = abs((-b - np.lib.scimath.sqrt(b**2 - 4*a*c))
                                / (2*a))
                    print("Criteria: ", max(crit1, crit2))
                y_old = 1

                times = []
                states = []
                states.append(y_old)
                exact_states = []
                exact_states.append(y_old)

                t = t_start
                times.append(t)
                step = 0
                rhs_hist = np.empty(order, dtype=complex)
                rhsns_hist = np.empty(order, dtype=complex)
                state_hist = np.empty(order, dtype=complex)
                time_hist = np.empty(order)
                rhs_hist[0] = stiff(y_old, mu)
                rhsns_hist[0] = nonstiff(y_old, lbda)
                state_hist[0] = y_old
                time_hist[0] = t
                tiny = 1e-15
                while t < t_end - tiny:
                    if step < order-1:
                        # "Bootstrap" using known exact solution.
                        # Substep to fill fast/nonstiff rhs hist.
                        y = exact(t + dt, mu, lbda)
                        dy_ns = nonstiff(y, lbda)
                        dy = stiff(y, mu)
                    else:
                        # Step normally - we have all the history we need.
                        y, dy_ns, dy = imex_mr_bdf(y_old, dt, t, lbda, mu,
                                                   state_hist,
                                                   rhs_hist,
                                                   rhsns_hist, time_hist,
                                                   order, sr)
                    # Rotate histories.
                    for i in range(order-1, 0, -1):
                        rhs_hist[i] = rhs_hist[i-1]
                        rhsns_hist[i] = rhsns_hist[i-1]
                        state_hist[i] = state_hist[i-1]
                        time_hist[i] = time_hist[i-1]
                    rhs_hist[0] = dy
                    rhsns_hist[0] = dy_ns
                    state_hist[0] = y
                    time_hist[0] = t + dt
                    # Append to states and prepare for next step.
                    states.append(y)
                    t += dt
                    times.append(t)
                    ey = exact(t, mu, lbda)
                    exact_states.append(ey)
                    y_old = y
                    step += 1

                # plt.clf()
                # plt.plot(times, states0, 'g-', times, exact_states0, 'k-',
                #          times, states1, 'b-', times, exact_states1, 'r-')
                # plt.legend(['y0', 'y0 Exact', 'y1', 'y1 Exact'])
                # plt.xlabel('t')
                # plt.ylabel('y')
                # plt.show()

                errors[j, k] = np.linalg.norm(y - exact(t, mu, lbda))
                eocrec.add_data_point(dt, errors[j, k])

            print("------------------------------------------------------")
            print("expected order: ", order)
            print("step ratio: ", sr)
            print("------------------------------------------------------")
            print(eocrec.pretty_print())

            orderest = eocrec.estimate_order_of_convergence()[0, 1]
            est_orders[j, sr_i] = orderest
            print("Estimated order of accuracy: ", orderest)

        if plot_runs:
            plot_individual_run(dts, errors, sr)

    plt.clf()
    plt.scatter(np.array(srs),
                np.array(est_orders[0, :]), c="g")
    plt.plot(srs, est_orders[0, :], "g--")
    plt.scatter(np.array(srs),
                np.array(est_orders[1, :]), c="b")
    plt.plot(srs, est_orders[1, :], "b--")
    plt.scatter(np.array(srs),
                np.array(est_orders[2, :]), c="r")
    plt.plot(srs, est_orders[2, :], "r--")
    plt.scatter(np.array(srs),
                np.array(est_orders[3, :]), c="k")
    plt.plot(srs, est_orders[3, :], "k--")
    plt.legend(["EOCs, Order 1", "EOCs, Order 2",
                "EOCs, Order 3", "EOCs, Order 4",
                "Order 1 Data", "Order 2 Data",
                "Order 3 Data", "Order 4 Data"])
    plt.xlabel("Step Ratio")
    plt.ylabel("EOC")
    plt.title("EOCs for Multi-Rate SBDF, Kaps' Problem")
    plt.show()


if __name__ == "__main__":
    main()
