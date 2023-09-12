import os
import h5py as h
import numpy as np
from astropy import units as u
from astropy import constants as const

#############################################
# Functions to calculate the Temperature from the ionization parameter
#############################################


def xiT(x, b, c, d, m):
    T = d + (1.5 - d) / (1 + (x / c) ** b) ** m
    return T


def interpPicog29(x, b, c, d, m):
    return d + (1.5 - d) / (1.0 + (x / c) ** b) ** m


def interpPicog30(x, b, c, d, m):
    return d + (1.6 - d) / (1.0 + (x / c) ** b) ** m


def interpPicog31(x, b, c, d, m):
    return d + (1.7 - d) / (1.0 + (x / c) ** b) ** m

#############################################
# Functions to read PLUTO outputs
#############################################


def getFilenames():
    filenames = [x for x in os.listdir("./") if ".h5" in x]
    filenames.sort()
    return filenames


def getVar(filename, step, variable):
    h5 = h.File(filename, "r")
    returnData = h5["Timestep_" + str(step) + "/vars"][variable][:]
    h5.close()
    return returnData


def getGridCell(filename=None, all=1):
    if not (filename):
        filename = getFilenames()[0]
    h5 = h.File(filename, "r")
    if all:
        x = h5["cell_coords"]["X"][:]
        y = h5["cell_coords"]["Y"][:]
        z = h5["cell_coords"]["Z"][:]
    else:
        x = h5["cell_coords"]["X"]
        y = h5["cell_coords"]["Y"]
        z = h5["cell_coords"]["Z"]
    x = x.astype("float64")
    y = y.astype("float64")
    z = z.astype("float64")
    return x, y, z


def remove_disc(T, Var, init):

    Var_new = np.copy(Var)
    b, a = np.shape(T)
    lgT = np.log10(T)
    Th_cut = np.zeros(a, dtype=int)
    for i in range(init, a):
        dlgT = np.diff(lgT[:, i])
        index_cut = dlgT.argmin()  # this is index of first cell in disc
        Var_new[index_cut:, i] = -1.
        Th_cut[i] = index_cut

    return Var_new, Th_cut


def calc_streamline_old(rmax, start_x, start_z, fvx, fvz):
    # fvx and fvz are functions that evaluate vx and vz at a given point.
    def diff_eqs(x, z):
        vz = fvz((x, z))
        vx = fvx((x, z))

        indomain = True

        if (abs(vz) < 10):
            indomain = False
        if (abs(vx) < 10):
            indomain = False

        return vz/vx, indomain

    xstream = []
    zstream = []
    xstream.append(start_x)
    zstream.append(start_z)

    in_domain = True
    counter = 0
    while in_domain:
        # calculate step-size based on local resolution
        radius = np.sqrt(xstream[-1]**2.+zstream[-1]**2.)
        i_r = (abs(rmax-radius)).argmin()
        # now calcuate dx, with correct sign
        dx = -np.sign(fvx((xstream[-1], zstream[-1])))*(rmax[i_r]-rmax[i_r-1])/10.
        # now calculate RK co-effients
        k1, in_domain = diff_eqs(xstream[-1], zstream[-1])
        k2, in_domain = diff_eqs(xstream[-1]+dx/2., zstream[-1]+dx/2.*k1)
        k3, in_domain = diff_eqs(xstream[-1]+dx/2., zstream[-1]+dx/2.*k2)
        k4, in_domain = diff_eqs(xstream[-1]+dx, zstream[-1]+dx*k3)

        if in_domain:
            zstream.append(zstream[-1]+dx/6.*(k1+2.*k2+2.*k3+k4))
            xstream.append(xstream[-1]+dx)

        counter += 1
        if (counter > 5000):
            break

    return xstream, zstream


def calc_streamline(rmax, start_x, start_z, fvx, fvz):

    in_domain = True

    # fvx and fvz are functions that evaluate vx and vz at a given point.
    def diff_eqs(x, z):
        vz = fvz((x, z))
        vx = fvx((x, z))

        if (abs(vz) < 10):
            in_domain = False
        if (abs(vx) < 10):
            in_domain = False

        return vz/vx

    # Coefficients used to compute the independent variable argument of f
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    # Coefficients used to compute the dependent variable argument of f
    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate
    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5

    # Coefficients related to the truncation error
    # Obtained through the difference of the 5th and 4th order RK methods:
    #     R = (1/h)|y5_i+1 - y4_i+1|
    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    # Init vectors to be returned
    xstream = []
    zstream = []
    P = []
    hmax = 0.1
    tol = 0.01
    inLower = 0.01
    inHigher = 0.1
    h = hmax
    xstream.append(start_x)
    zstream.append(start_z)
    counter = 0

    while in_domain:
        # calculate step-size based on local resolution
        radius = np.sqrt(xstream[-1]**2.+zstream[-1]**2.)
        i_r = (abs(rmax-radius)).argmin()
        # now calcuate dx, with correct sign
        h = -np.sign(fvx((xstream[-1], zstream[-1])))*(rmax[i_r]-rmax[i_r-1])/100.
        if counter == 0:
            P.append(h)

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.
        x = xstream[-1]
        y = zstream[-1]
        k1 = h * diff_eqs(x, y)
        k2 = h * diff_eqs(x + a2 * h, y + b21 * k1)
        k3 = h * diff_eqs(x + a3 * h, y + b31 * k1 + b32 * k2)
        k4 = h * diff_eqs(x + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
        k5 = h * diff_eqs(y + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
        k6 = h * diff_eqs(y + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)

        # Calulate local truncation error
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
        # If it is less than the tolerance, the step is accepted and RK4 value is stored
        if r <= tol:
            x = x + h
            y = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            xstream.append(x)
            zstream.append(y)
            P.append(h)

        # Prevent zero division
        if r == 0:
            r = P[-1]

        # Calculate next step size
        h = h * min(max(0.84 * (tol / r)**0.25, inLower), inHigher)

        # Upper limit with hmax and lower with hmin
        # h = hmax if h > hmax else hmin if h < hmin else h
        counter += 1

    return xstream, zstream  # , Ts, rhos, vphis

#############################################
# Functions defining the mass-loss rates
#############################################


def MdotLxPicogna(Lx1, Lx2):
    AL = -2.7326
    BL = 3.3307
    CL = -2.9868e-3
    DL = -7.2580
    mdot1 = 10 ** (AL * np.exp(((np.log(np.log10(Lx1)) - BL) ** 2) / CL) + DL)
    mdot2 = 10 ** (AL * np.exp(((np.log(np.log10(Lx2)) - BL) ** 2) / CL) + DL)
    return mdot1 / mdot2


def Sigmadot(a, b, c, d, e, f, g, Rau):
    logR = np.log10(Rau)
    lnx = np.log(Rau)
    ln10 = np.log(10)
    return (
        ln10
        * (
            6 * a * lnx ** 5 / (Rau * ln10 ** 6)
            + 5 * b * lnx ** 4 / (Rau * ln10 ** 5)
            + 4 * c * lnx ** 3 / (Rau * ln10 ** 4)
            + 3 * d * lnx ** 2 / (Rau * ln10 ** 3)
            + 2 * e * lnx / (Rau * ln10 ** 2)
            + f / (Rau * ln10)
        )
        * 10
        ** (
            a * logR ** 6
            + b * logR ** 5
            + c * logR ** 4
            + d * logR ** 3
            + e * logR ** 2
            + f * logR
            + g
        )
    )


def Lx(*Mstar):
    for x in Mstar:
        maxLx = 10 ** (1.42 * np.log10(x) + 30.37)
        minLx = 10 ** (1.66 * np.log10(x) + 30.25)
        aveLx = 10 ** (1.54 * np.log10(x) + 30.31)
        return minLx, aveLx, maxLx


#############################################
# Miscellaneous
#############################################

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10 ** exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

