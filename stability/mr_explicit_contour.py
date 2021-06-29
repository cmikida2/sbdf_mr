import numpy as np


def main():

    # Determining max(z) contours for each SR in first
    # order MR.
    order = 1
    sr = 5
    # Modify this to see what effect stiffness has on
    # the stability of the explicit part of the SBDF.
    mu = 0

    # root locus: loop through theta.
    imags = np.linspace(-4.5, 4.5, 500)
    reals = np.linspace(-sr - 4, 1, 500)
    zs = np.zeros((500, 500))

    # Now, calculate the stability bound for a
    # single part of the IMEX scheme.
    for i, re in enumerate(reals):
        print("i = ", i)
        for j, im in enumerate(imags):

            lbda = re + 1j*im
            if order == 1:
                ynt = (1 + lbda) / (1 - mu)
                ynp = 1 + (1/sr)*lbda + (1/sr)*mu*ynt
                # Substep.
                for _k in range(2, sr + 1):
                    ynp = ynp*(1 + (1/sr)*lbda) + (1/sr)*mu*ynt
                crit = (abs(1 + lbda*ynp)/(1 - mu))
                zs[j, i] = crit
            elif order == 2:
                a = 3/2 - mu
                b = -(2*lbda + 2)
                c = lbda + 1/2
                crit1 = abs((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
                crit2 = abs((-b - np.sqrt(b**2 - 4*a*c)) / (2*a))
                zs[j, i] = max(crit1, crit2)
            elif order == 3:
                # Order 3 has a very complicated polynomial
                crit = max(abs(np.roots([11/6 - mu, -3*(lbda + 1),
                                        3*(lbda + 1/2), -(lbda + 1/3)])))
                zs[j, i] = crit
            else:
                # Order 4 has a very complicated polynomial
                crit = max(abs(np.roots([1 - (12/25)*mu, -(48/25)*(lbda + 1),
                                         (36/25)*(2*lbda + 1),
                                         -(16/25)*(3*lbda + 1),
                                         (3/25)*(4*lbda + 1)])))
                zs[j, i] = crit

    import matplotlib.pyplot as plt
    # Contour plot of the maximum amplification factor
    plt.clf()
    plt.contourf(reals, imags, zs, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                           0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,
                                           1.4, 1.5])
    plt.title("SBDF MR Order {order} Maximum Amplification "
              "Factor, Explicit Case, SR = {sr}".format(order=order, sr=sr))
    plt.xlabel("Re(lambda)")
    plt.ylabel("Im(lambda)")
    plt.colorbar(label="z")
    plt.contour(reals, imags, zs, levels=[0, 1], colors=['red'])
    plt.show()


if __name__ == "__main__":
    main()
