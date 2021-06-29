import numpy as np
import matplotlib.pyplot as plt


def kaps_nonstiff(y, eps):

    # Kaps' first problem - nonstiff part.
    dy = np.zeros(2)
    dy[0] = -2*y[0]
    dy[1] = y[0] - y[1] - y[1]*y[1]
    return dy


def kaps_stiff(y, eps):

    # Kaps' first problem - stiff part.
    einv = 1/eps
    dy = np.zeros(2)
    dy[0] = -einv*y[0] + einv*y[1]*y[1]
    dy[1] = 0
    return dy


def kaps_exact(t):
    y_exact = np.zeros(2)
    y_exact[1] = np.exp(-t)
    y_exact[0] = y_exact[1]*y_exact[1]
    return y_exact


def nonstiff_extrap(y, h, eps, rhsns_hist, state_hist, order, rhs_extrap,
                    sr, subst_i):

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

        return h*kaps_nonstiff(y_new, eps)
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


def lagrange(t, hist, time_hist, order):

    a_new = 0
    for i in range(0, order):
        term = hist[i]
        for j in range(0, order):
            if i != j:
                term = term*(t - time_hist[j])/(time_hist[i] - time_hist[j])
        a_new += term

    return a_new


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


def imex_mr_bdf(y, h, t, eps, state_hist, rhs_hist, rhsns_hist, order, sr):

    # "Slowest first:" - first, use RHS extrapolation to form implicit
    # prediction, then solve.
    nonstiff_cont = nonstiff_extrap(y, h, eps, rhsns_hist, state_hist, order,
                                    True, sr, sr)
    y_new = np.zeros(2)
    einv = 1/eps

    # Stiff contribution to y2 is zero, so we can do an implicit solve
    # directly without any Newton nonsense...
    if order == 1:
        y_new[1] = state_hist[0, 1] + nonstiff_cont[1]
        y_new[0] = (1/(1 + h*einv))*(state_hist[0, 0] +
                                     h*einv*y_new[1]*y_new[1] +
                                     nonstiff_cont[0])
    elif order == 2:
        y_new[1] = (4.0/3.0)*state_hist[0, 1] - (1.0/3.0)*state_hist[1, 1] + \
                (2.0/3.0)*nonstiff_cont[1]
        y_new[0] = (1/(1 + (2.0/3.0)*h*einv))*((4.0/3.0)*state_hist[0, 0] -
                                               (1.0/3.0)*state_hist[1, 0] +
                                               (2.0/3.0) * h * einv *
                                               y_new[1]*y_new[1] +
                                               (2.0/3.0)*nonstiff_cont[0])
    elif order == 3:
        y_new[1] = (18.0/11.0)*state_hist[0, 1] - \
                (9.0/11.0)*state_hist[1, 1] + \
                (2.0/11.0)*state_hist[2, 1] + \
                (6.0/11.0)*nonstiff_cont[1]
        y_new[0] = (1/(1 + (6.0/11.0)*h*einv))*((18.0/11.0)*state_hist[0, 0] -
                                                (9.0/11.0)*state_hist[1, 0] +
                                                (2.0/11.0)*state_hist[2, 0] +
                                                (6.0/11.0) * h * einv *
                                                y_new[1]*y_new[1] +
                                                (6.0/11.0)*nonstiff_cont[0])
    else:
        y_new[1] = (48.0/25.0)*state_hist[0, 1] - \
                (36.0/25.0)*state_hist[1, 1] + \
                (16.0/25.0)*state_hist[2, 1] - \
                (3.0/25.0)*state_hist[3, 1] + \
                (12.0/25.0)*nonstiff_cont[1]
        y_new[0] = (1/(1 + (12.0/25.0)*h*einv))*((48.0/25.0)*state_hist[0, 0] -
                                                 (36.0/25.0)*state_hist[1, 0] +
                                                 (16.0/25.0)*state_hist[2, 0] -
                                                 (3.0/25.0)*state_hist[3, 0] +
                                                 (12.0/25.0) * h * einv *
                                                 y_new[1]*y_new[1] +
                                                 (12.0/25.0)*nonstiff_cont[0])

    new_stiff = kaps_stiff(y_new, eps)

    # Now do AB at the substep level, supplementing with interpolation of the
    # newfound stiff RHS.
    if sr > 1:
        # For the first substep, use existing RHS history...
        substep_rhs_hist = rhsns_hist.copy() + rhs_hist.copy()
        substep_time_hist = np.zeros(order)
        substep_state_hist = np.zeros((order, 2))
        for i in range(0, order):
            substep_time_hist[i] = t - i*h

        y_substep = adams_bashforth(y, (1/sr)*h, order,
                                    substep_time_hist, substep_rhs_hist,
                                    substep_time_hist[0])

        # Evaluate to obtain nonstiff RHS.
        nonstiff_rhs = kaps_nonstiff(y_substep, eps)

        # Build nonstiff RHS history and accompanying
        # time history.
        new_rhsns_hist = np.zeros((order, 2))
        new_rhsns_thist = np.zeros(order)
        for i in range(0, order-1):
            new_rhsns_hist[i+1] = rhsns_hist[i]
            new_rhsns_thist[i+1] = t - i*h
            substep_state_hist[i+1] = state_hist[i]
        new_rhsns_hist[0] = nonstiff_rhs
        new_rhsns_thist[0] = t + (1/sr)*h
        substep_state_hist[0] = y_substep

        if sr > 2:
            # Now build a stiff RHS history so we can interpolate with it.
            new_rhs_hist = np.zeros((order+1, 2))
            for i in range(0, order):
                new_rhs_hist[i+1] = rhs_hist[i]
            new_rhs_hist[0] = new_stiff

            # Interpolate to obtain stiff RHS.
            stiff_rhs = stiff_interp((1/sr), new_rhs_hist, order)
            # Combine and rotate substep-level history and time history.
            # Rotate this into history.
            for i in range(order-1, 0, -1):
                substep_rhs_hist[i] = substep_rhs_hist[i-1]
                substep_time_hist[i] = substep_time_hist[i-1]
            substep_rhs_hist[0] = nonstiff_rhs + stiff_rhs
            substep_time_hist[0] = t + (1/sr)*h

            for j in range(2, sr):
                y_substep = adams_bashforth(y_substep, (1/sr)*h, order,
                                            substep_time_hist,
                                            substep_rhs_hist,
                                            substep_time_hist[0])
                # Interpolate to obtain stiff RHS.
                stiff_rhs = stiff_interp((j/sr), new_rhs_hist, order)
                # Evaluate to obtain nonstiff RHS.
                nonstiff_rhs = kaps_nonstiff(y_substep, eps)
                # Combine and rotate substep-level history and time history.
                # Rotate this into history.
                for i in range(order-1, 0, -1):
                    substep_rhs_hist[i] = substep_rhs_hist[i-1]
                    substep_time_hist[i] = substep_time_hist[i-1]
                    substep_state_hist[i] = substep_state_hist[i-1]
                    new_rhsns_hist[i] = new_rhsns_hist[i-1]
                    new_rhsns_thist[i] = new_rhsns_thist[i-1]
                substep_rhs_hist[0] = nonstiff_rhs + stiff_rhs
                substep_time_hist[0] = t + (j/sr)*h
                substep_state_hist[0] = y_substep
                new_rhsns_thist[0] = t + (j/sr)*h
                new_rhsns_hist[0] = nonstiff_rhs

        # FIXME: instead of evaluating, we do the same history-to-prediction
        # thing as standard BDF...but we need a new nonstiff RHS history.
        # FIXME: state or RHS extrap?
        # new_nonstiff = kaps_nonstiff(y_substep, eps)
        nonstiff_cont = nonstiff_extrap(y, h, eps, new_rhsns_hist,
                                        substep_state_hist, order, True, sr,
                                        sr)

        # Re-solve.
        if order == 1:
            y_new[1] = substep_state_hist[0, 1] + (1/sr)*nonstiff_cont[1]
            y_new[0] = (1/(1 + (1/sr)*h*einv))*(substep_state_hist[0, 0] +
                                                (1/sr)*h*einv*y_new[1]*y_new[1]
                                                + (1/sr)*nonstiff_cont[0])
        elif order == 2:
            y_new[1] = (4.0/3.0)*substep_state_hist[0, 1] - \
                    (1.0/3.0)*substep_state_hist[1, 1] + \
                    (2.0/3.0)*(1/sr)*nonstiff_cont[1]
            y_new[0] = (1/(1 + (1/sr)*(2.0/3.0)*h*einv)) * \
                ((4.0/3.0)*substep_state_hist[0, 0] -
                 (1.0/3.0)*substep_state_hist[1, 0] +
                 (1/sr)*(2.0/3.0) * h * einv *
                 y_new[1]*y_new[1] +
                 (1/sr)*(2.0/3.0)*nonstiff_cont[0])
        elif order == 3:
            y_new[1] = (18.0/11.0)*state_hist[0, 1] - \
                    (9.0/11.0)*state_hist[1, 1] + \
                    (2.0/11.0)*state_hist[2, 1] + \
                    (6.0/11.0)*nonstiff_cont[1]
            y_new[0] = (1/(1 + (6.0/11.0)*h*einv)) * \
                ((18.0/11.0)*state_hist[0, 0] -
                 (9.0/11.0)*state_hist[1, 0] +
                 (2.0/11.0)*state_hist[2, 0] +
                 (6.0/11.0) * h * einv *
                 y_new[1]*y_new[1] +
                 (6.0/11.0)*nonstiff_cont[0])
        else:
            y_new[1] = (48.0/25.0)*state_hist[0, 1] - \
                    (36.0/25.0)*state_hist[1, 1] + \
                    (16.0/25.0)*state_hist[2, 1] - \
                    (3.0/25.0)*state_hist[3, 1] + \
                    (12.0/25.0)*nonstiff_cont[1]
            y_new[0] = (1/(1 + (12.0/25.0)*h*einv)) * \
                ((48.0/25.0)*state_hist[0, 0] -
                 (36.0/25.0)*state_hist[1, 0] +
                 (16.0/25.0)*state_hist[2, 0] -
                 (3.0/25.0)*state_hist[3, 0] +
                 (12.0/25.0) * h * einv *
                 y_new[1]*y_new[1] +
                 (12.0/25.0)*nonstiff_cont[0])

    # FIXME: correction on nonstiff term?
    new_nonstiff = kaps_nonstiff(y_new, eps)
    # FIXME: incorporate re-solve?
    new_stiff = kaps_stiff(y_new, eps)
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

    # Time integration of Kaps' problem to test
    # some new IMEX methods.
    t_start = 0
    t_end = 1
    dts = [0.05, 0.01, 0.005, 0.001]
    # dts = [0.01, 0.005, 0.001, 0.0005]
    orders = [2]
    # orders = [3]
    errors = np.zeros((4, 4))
    srs = [1, 2, 3, 4, 5]
    # srs = [1, 5, 10, 50, 100]
    # srs = [2]
    est_orders = np.zeros((4, 5))
    eps = 0.001
    plot_runs = False

    for sr_i, sr in enumerate(srs):
        for j, order in enumerate(orders):
            from pytools.convergence import EOCRecorder
            eocrec = EOCRecorder()
            for k, dt in enumerate(dts):

                y_old = np.zeros(2)
                y_old[0] = 1
                y_old[1] = 1

                times = []
                states0 = []
                states0.append(y_old[0])
                states1 = []
                states1.append(y_old[1])
                exact_states0 = []
                exact_states1 = []
                exact_states0.append(y_old[0])
                exact_states1.append(y_old[1])

                t = t_start
                times.append(t)
                step = 0
                rhs_hist = np.empty((order, 2), dtype=y_old.dtype)
                rhsns_hist = np.empty((order, 2), dtype=y_old.dtype)
                state_hist = np.empty((order+1, 2), dtype=y_old.dtype)
                rhs_hist[0] = kaps_stiff(y_old, eps)
                rhsns_hist[0] = kaps_nonstiff(y_old, eps)
                state_hist[0] = y_old
                tiny = 1e-15
                while t < t_end - tiny:
                    if step < order:
                        # "Bootstrap" using known exact solution.
                        # Substep to fill fast/nonstiff rhs hist.
                        y = kaps_exact(t + dt)
                        dy_ns = kaps_nonstiff(y, eps)
                        dy = kaps_stiff(y, eps)
                    else:
                        # Step normally - we have all the history we need.
                        y, dy_ns, dy = imex_mr_bdf(y_old, dt, t, eps,
                                                   state_hist,
                                                   rhs_hist,
                                                   rhsns_hist, order,
                                                   sr)
                    # Rotate histories.
                    for i in range(order-1, 0, -1):
                        rhs_hist[i] = rhs_hist[i-1]
                        rhsns_hist[i] = rhsns_hist[i-1]
                    for i in range(order, 0, -1):
                        state_hist[i] = state_hist[i-1]
                    rhs_hist[0] = dy
                    rhsns_hist[0] = dy_ns
                    state_hist[0] = y
                    # Append to states and prepare for next step.
                    states0.append(y[0])
                    states1.append(y[1])
                    t += dt
                    times.append(t)
                    ey = kaps_exact(t)
                    exact_states0.append(ey[0])
                    exact_states1.append(ey[1])
                    y_old = y
                    step += 1

                # plt.clf()
                # plt.plot(times, states0, 'g-', times, exact_states0, 'k-',
                #          times, states1, 'b-', times, exact_states1, 'r-')
                # plt.legend(['y0', 'y0 Exact', 'y1', 'y1 Exact'])
                # plt.xlabel('t')
                # plt.ylabel('y')
                # plt.show()

                errors[j, k] = np.linalg.norm(y - kaps_exact(t))
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
    plt.title("EOCs for Multi-Rate SBDF, Kaps' Problem, High SR")
    plt.show()


if __name__ == "__main__":
    main()
