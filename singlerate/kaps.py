import numpy as np


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


def ab_nonstiff(y, h, eps, rhsns_hist, state_hist, order, rhs_extrap):

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

        return h*kaps_nonstiff(y_new, eps)
    else:
        # Option 2: use RHS histories instead, a la AB.
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


def imex_bdf(y, h, eps, state_hist, rhs_hist, rhsns_hist, order, rhs_extrap):

    # For now, avoid doing any predicting,
    # as we don't need (or want) to worry
    # about step size or order changes.

    nonstiff_cont = ab_nonstiff(y, h, eps, rhsns_hist,
                                state_hist, order, rhs_extrap)
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

    if rhs_extrap:
        return (y_new, kaps_nonstiff(y_new, eps),
                kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))
    else:
        # no correction on nonstiff term?
        # return (y_new, nonstiff_cont/h, nonstiff_cont/h +
        #         kaps_stiff(y_new, eps))
        new_nonstiff = kaps_nonstiff(y_new, eps)
        return (y_new, new_nonstiff,
                new_nonstiff + kaps_stiff(y_new, eps))


def bdf(y, h, eps, state_hist, rhs_hist, rhsns_hist, order):

    # For now, avoid doing any predicting,
    # as we don't need (or want) to worry
    # about step size or order changes.

    # Use Scipy root here
    from scipy.optimize import root
    if order == 1:
        y_new = root(lambda unk: unk - state_hist[0] -
                     h*(kaps_stiff(unk, eps) +
                        kaps_nonstiff(unk, eps)), y).x
    elif order == 2:
        y_new = root(lambda unk: unk - (4.0/3.0)*state_hist[0] +
                     (1.0/3.0)*state_hist[1] -
                     (2.0/3.0)*h*(kaps_stiff(unk, eps) +
                                  kaps_nonstiff(unk, eps)), y).x
    elif order == 3:
        y_new = root(lambda unk: unk - (18.0/11.0)*state_hist[0] +
                     (9.0/11.0)*state_hist[1] - (2.0/11.0)*state_hist[2] -
                     (6.0/11.0)*h*(kaps_stiff(unk, eps) +
                                   kaps_nonstiff(unk, eps)), y).x
    else:
        y_new = root(lambda unk: unk - (48.0/25.0)*state_hist[0] +
                     (36.0/25.0)*state_hist[1] - (16.0/25.0)*state_hist[2] +
                     (3.0/25.0)*state_hist[3] -
                     (12.0/25.0)*h*(kaps_stiff(unk, eps) +
                                    kaps_nonstiff(unk, eps)), y).x

    return (y_new, kaps_nonstiff(y_new, eps),
            kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))


def adams_bashforth(y, h, eps, order, state_hist, rhs_hist):

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

    return (y_new, kaps_nonstiff(y_new, eps),
            kaps_nonstiff(y_new, eps) + kaps_stiff(y_new, eps))


def main():

    # Time integration of Kaps' problem to test
    # some new IMEX methods.
    t_start = 0
    t_end = 1
    # dts = [0.05, 0.01, 0.005, 0.001]
    dts = [0.2, 0.1, 0.05, 0.01]
    orders = [1, 2, 3, 4]
    errors = np.zeros((4, 4))
    errorsr = np.zeros((4, 4))
    stepper = 'sbdf'
    eps = 0.0001

    from pytools.convergence import EOCRecorder
    for j, order in enumerate(orders):
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
            rhs_hist[0] = (kaps_stiff(y_old, eps) + kaps_nonstiff(y_old, eps))
            rhsns_hist[0] = kaps_nonstiff(y_old, eps)
            state_hist[0] = y_old
            tiny = 1e-15
            while t < t_end - tiny:
                if step < order:
                    # "Bootstrap" using known exact solution.
                    y = kaps_exact(t + dt)
                    dy_ns = kaps_nonstiff(y, eps)
                    dy_full = dy_ns + kaps_stiff(y, eps)
                else:
                    # Step normally - we have all the history we need.
                    if stepper == 'sbdf':
                        y, dy_ns, dy_full = imex_bdf(y_old, dt, eps,
                                                     state_hist, rhs_hist,
                                                     rhsns_hist, order, False)
                    elif stepper == 'bdf':
                        y, dy_ns, dy_full = bdf(y_old, dt, eps, state_hist,
                                                rhs_hist, rhsns_hist, order)
                    else:
                        y, dy_ns, dy_full = adams_bashforth(y_old, dt, eps,
                                                            order, state_hist,
                                                            rhs_hist)
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
                states0.append(y[0])
                states1.append(y[1])
                t += dt
                times.append(t)
                ey = kaps_exact(t)
                exact_states0.append(ey[0])
                exact_states1.append(ey[1])
                y_old = y
                step += 1

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.plot(times, states0, 'g-', times, exact_states0, 'k-',
            #          times, states1, 'b-', times, exact_states1, 'r-')
            # plt.legend(['y0','y0 Exact', 'y1', 'y1 Exact'])
            # plt.xlabel('t')
            # plt.ylabel('y')
            # plt.show()

            # plt.clf()
            # plt.plot(times, (np.array(states0) - np.array(exact_states0)),
            #          'g-', times,
            #          (np.array(states1) - np.array(exact_states1)), 'r-')
            # plt.legend(['y0 Error','y1 Error'], 2)
            # plt.xlabel('t')
            # plt.ylabel('error')
            # plt.show()

            errors[j, k] = np.linalg.norm(y - kaps_exact(t))
            eocrec.add_data_point(dt, errors[j, k])

        print("------------------------------------------------------")
        print("expected order: ", order)
        print("------------------------------------------------------")
        print(eocrec.pretty_print())

        orderest = eocrec.estimate_order_of_convergence()[0, 1]
        print("Estimated order of accuracy: ", orderest)

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
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[0, :]))), c='g')
    plt.plot(np.log10(np.array(dts)), p1(np.log10(np.array(dts))), 'g--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[1, :]))), c='b')
    plt.plot(np.log10(np.array(dts)), p2(np.log10(np.array(dts))), 'b--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[2, :]))), c='r')
    plt.plot(np.log10(np.array(dts)), p3(np.log10(np.array(dts))), 'r--')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errors[3, :]))), c='k')
    plt.plot(np.log10(np.array(dts)), p4(np.log10(np.array(dts))), 'k--')
    # plt.legend(["EOC=%.6f" % (z1[0]), "EOC=%.6f" % (z2[0]),
    #             "EOC=%.6f" % (z3[0]), "EOC=%.6f" % (z4[0]),
    #             'Order 1 Data, State', 'Order 2 Data, State',
    #             'Order 3 Data, State', 'Order 4 Data, State'])
    plt.xlabel('log10(dt)')
    plt.ylabel('log10(err)')
    plt.ylim((-16, -1))
    plt.title('Log-Log Error Timestep Plots: SBDF')
    # plt.show()

    # Now the RHS extrapolation.
    for j, order in enumerate(orders):
        eocrecr = EOCRecorder()
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
            rhs_hist[0] = (kaps_stiff(y_old, eps) + kaps_nonstiff(y_old, eps))
            rhsns_hist[0] = kaps_nonstiff(y_old, eps)
            state_hist[0] = y_old
            tiny = 1e-15
            while t < t_end - tiny:
                if step < order:
                    # "Bootstrap" using known exact solution.
                    y = kaps_exact(t + dt)
                    dy_ns = kaps_nonstiff(y, eps)
                    dy_full = dy_ns + kaps_stiff(y, eps)
                else:
                    # Step normally - we have all the history we need.
                    if stepper == 'sbdf':
                        y, dy_ns, dy_full = imex_bdf(y_old, dt, eps,
                                                     state_hist, rhs_hist,
                                                     rhsns_hist, order, True)
                    elif stepper == 'bdf':
                        y, dy_ns, dy_full = bdf(y_old, dt, eps, state_hist,
                                                rhs_hist, rhsns_hist, order)
                    else:
                        y, dy_ns, dy_full = adams_bashforth(y_old, dt, eps,
                                                            order, state_hist,
                                                            rhs_hist)
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
                states0.append(y[0])
                states1.append(y[1])
                t += dt
                times.append(t)
                ey = kaps_exact(t)
                exact_states0.append(ey[0])
                exact_states1.append(ey[1])
                y_old = y
                step += 1

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.plot(times, states0, 'g-', times, exact_states0, 'k-',
            #          times, states1, 'b-', times, exact_states1, 'r-')
            # plt.legend(['y0','y0 Exact', 'y1', 'y1 Exact'])
            # plt.xlabel('t')
            # plt.ylabel('y')
            # plt.show()

            # plt.clf()
            # plt.plot(times, (np.array(states0) - np.array(exact_states0)),
            #          'g-', times,
            #          (np.array(states1) - np.array(exact_states1)), 'r-')
            # plt.legend(['y0 Error','y1 Error'], 2)
            # plt.xlabel('t')
            # plt.ylabel('error')
            # plt.show()

            errorsr[j, k] = np.linalg.norm(y - kaps_exact(t))
            eocrecr.add_data_point(dt, errorsr[j, k])

        print("------------------------------------------------------")
        print("expected order: ", order)
        print("------------------------------------------------------")
        print(eocrecr.pretty_print())

        orderest = eocrecr.estimate_order_of_convergence()[0, 1]
        print("Estimated order of accuracy: ", orderest)

    z1r = np.polyfit(np.log10(np.array(dts)),
                     np.log10(np.abs(np.array(errorsr[0, :]))), 1)
    z2r = np.polyfit(np.log10(np.array(dts)),
                     np.log10(np.abs(np.array(errorsr[1, :]))), 1)
    z3r = np.polyfit(np.log10(np.array(dts)),
                     np.log10(np.abs(np.array(errorsr[2, :]))), 1)
    z4r = np.polyfit(np.log10(np.array(dts)),
                     np.log10(np.abs(np.array(errorsr[3, :]))), 1)
    p1r = np.poly1d(z1r)
    p2r = np.poly1d(z2r)
    p3r = np.poly1d(z3r)
    p4r = np.poly1d(z4r)
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errorsr[0, :]))), c='g', marker='v')
    plt.plot(np.log10(np.array(dts)), p1r(np.log10(np.array(dts))), 'g-')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errorsr[1, :]))), c='b', marker='v')
    plt.plot(np.log10(np.array(dts)), p2r(np.log10(np.array(dts))), 'b-')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errorsr[2, :]))), c='r', marker='v')
    plt.plot(np.log10(np.array(dts)), p3r(np.log10(np.array(dts))), 'r-')
    plt.scatter(np.log10(np.array(dts)),
                np.log10(np.abs(np.array(errorsr[3, :]))), c='k', marker='v')
    plt.plot(np.log10(np.array(dts)), p4r(np.log10(np.array(dts))), 'k-')
    plt.legend(["State EOC=%.6f" % (z1[0]), "State EOC=%.6f" % (z2[0]),
                "State EOC=%.6f" % (z3[0]), "State EOC=%.6f" % (z4[0]),
                "RHS EOC=%.6f" % (z1r[0]), "RHS EOC=%.6f" % (z2r[0]),
                "RHS EOC=%.6f" % (z3r[0]), "RHS EOC=%.6f" % (z4r[0]),
                "Order 1 Data, State", "Order 2 Data, State",
                "Order 3 Data, State", "Order 4 Data, State",
                'Order 1 Data, RHS', 'Order 2 Data, RHS',
                'Order 3 Data, RHS', 'Order 4 Data, RHS'])
    plt.show()


if __name__ == "__main__":
    main()
