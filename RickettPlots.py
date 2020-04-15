import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from pint.models import get_model
from pint import toa
from matplotlib.ticker import MultipleLocator
from astropy.table import Table
from astropy import constants as const
from RickettEquations import K_coeffs, TISS, k_norm, RotateVector


plot_TISS = False  # toggle whether the interstellar scintillation timescale is plotted
plot_knorm_all = False  # toggle whether the normalized harmonic coefficients are plotted
# toggle normalized harmonic coefficients being plotted for increased
# velocities
plot_knorm_VIS_inc = False
# toggle normalized harmonic coefficients being plotted for decreased
# velocities
plot_knorm_VIS_dec = False


t_start = 52997  # 52997
t_end = 53561  # 53561
t_nsteps = 564  # 564
par = "J0737-3039A.par"

# -------------------------------------------------------------------------------------
# Fitted values from Table 3 of Rickett et al 2014
# -------------------------------------------------------------------------------------

fitval90 = {'i': 90. * u.deg,
            's': 0.71,
            'Oangle': 69 * u.deg,
            'R': 0.76,
            'PsiAR': 72 * u.deg,
            'VIS': np.array([-12,
                             50]) * u.km / u.s,
            's0': 4.2e6 * u.m,
            'dpc': 1150 * u.pc}

fitval88 = {'i': 88.7 * u.deg,
            's': 0.71,
            'Oangle': 61 * u.deg,
            'R': 0.71,
            'PsiAR': 61 * u.deg,
            'VIS': np.array([-9,
                             42]) * u.km / u.s,
            's0': 4.2e6 * u.m,
            'dpc': 1150 * u.pc}

fitval91 = {'i': 91.3 * u.deg,
            's': 0.70,
            'Oangle': 111 * u.deg,
            'R': 0.96,
            'PsiAR': 118 * u.deg,
            'VIS': np.array([-79,
                             100]) * u.km / u.s,
            's0': 4.2e6 * u.m,
            'dpc': 1150 * u.pc}

# ----------------------------------------
# Set up binary pulsar parameters	#
# ----------------------------------------

if (plot_TISS or plot_knorm_all or plot_knorm_VIS_inc or plot_knorm_VIS_dec):
    psr_m = get_model(par)  # Read in known values for par file (from ATNF)

    psr = SkyCoord(
        ra=str(
            psr_m.RAJ.quantity),
        dec=str(
            psr_m.DECJ.quantity),
        pm_ra_cosdec=psr_m.PMRA.quantity,
        pm_dec=psr_m.PMDEC.quantity,
        distance=fitval90['dpc'])

    t = toa.make_fake_toas(
        t_start,
        t_end,
        t_nsteps,
        psr_m,
        freq=820,
        obs="GBT")

    # NOTE: You can only get binary model values IFF you call psr_m.delay(t)
    # first!!!
    psr_m.delay(t)
    bm = psr_m.binary_instance

if (plot_TISS):

    ax = plt.subplot(111)
    plt.grid(linestyle='dotted')
    plt.xlabel('Orbital Phase (deg)')
    plt.ylabel('ISS Timescale (sec)')
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    plt.xlim(0, 360)

    phi = np.linspace(0, 360, 360)
    phi = np.radians(phi)

    fitval = fitval90

    fitval['s0'] = 6.1e6 * u.m

    # Calculate K coefficients
    Ks0 = K_coeffs(t_start, fitval, psr, psr_m)
    print('K Coefficients for i = 90', Ks0)

    TS_plot = np.ones(phi.size)

    for angle in phi:
        TS_plot[np.where(phi == angle)] = TISS(Ks0, angle)

    ax.plot(np.degrees(phi), TS_plot, label=r'i=90.0$^\circ$')

    fitval = fitval91

    fitval['s0'] = 9.3e6 * u.m

    Ks0 = K_coeffs(t_start, fitval, psr, psr_m)

    print('K Coefficients for i = 91.3', Ks0)

    TS_plot = np.ones(phi.size)

    for angle in phi:
        TS_plot[np.where(phi == angle)] = TISS(Ks0, angle)

    ax.plot(np.degrees(phi), TS_plot, label=r'i=91.3$^\circ$')

    fitval = fitval88
    fitval['s0'] = 5.7e6 * u.m

    Ks0 = K_coeffs(t_start, fitval, psr, psr_m)
    print('K Coefficients for i = 88.7', Ks0)

    TS_plot = np.ones(phi.size)

    for angle in phi:
        TS_plot[np.where(phi == angle)] = TISS(Ks0, angle)

    ax.plot(np.degrees(phi), TS_plot, label=r'i=88.7$^\circ$')

    tiss_data = Table.read(
        '53467.tiss5t',
        format='ascii.no_header',
        data_start=11,
        names=(
            'phi',
            'T_iss',
            'T_iss_error'))

    plt.errorbar(
        tiss_data['phi'],
        tiss_data['T_iss'],
        yerr=tiss_data['T_iss_error'],
        fmt='o',
        label='Data')
    ax.legend()
    plt.title('MJD 52997')
    plt.show()

if (plot_knorm_all or plot_knorm_VIS_inc or plot_knorm_VIS_dec):

    # observation days from Table 1
    obs_day = [-3, 211, 311, 379, 467, 202, 203, 274,
               312, 319, 374, 378, 415, 451, 462, 505, 560]

    # k_fit[0] is K0 (col. 1 of Table 1), k_fit[1] is Ks (col.2 of Table 1),
    # k_fit[2] is Kc2 (col. 5 of Table 1)
    k_fit = np.array([[3.46, 3.79, 4.76, 6.19, 3.25, 0.96, 0.89, 0.92, 1.35, 1.41, 0.91, 0.94, 0.74, 0.84, 0.74, 0.54, 0.29], [-2.63, 0.70, -3.63, -6.74, -1.57, 0.41, 0.29, -0.40, - \
                     1.23, -1.39, -1.02, -1.05, -0.80, -0.71, -0.60, 0.06, 0.16], [-2.77, -3.51, -4.02, -4.40, -2.57, -1.07, -0.86, -0.84, -1.15, -1.13, -0.59, -0.62, -0.53, -0.67, -0.58, -0.49, -0.26]])

    # k_err[0] is K0 (col. 1 of Table 1), k_err[1] is Ks (col.2 of Table 1),
    # k_err[2] is Kc2 (col. 5 of Table 1)
    k_err = np.array([[0.06,
                       0.10,
                       0.13,
                       0.14,
                       0.08,
                       0.09,
                       0.05,
                       0.06,
                       0.06,
                       0.05,
                       0.05,
                       0.03,
                       0.04,
                       0.03,
                       0.02,
                       0.03,
                       0.01],
                      [0.08,
                       0.11,
                       0.16,
                       0.19,
                       0.10,
                       0.15,
                       0.07,
                       0.08,
                       0.07,
                       0.07,
                       0.09,
                       0.05,
                       0.06,
                       0.05,
                       0.03,
                       0.04,
                       0.02],
                      [0.06,
                       0.10,
                       0.12,
                       0.12,
                       0.08,
                       0.12,
                       0.05,
                       0.06,
                       0.06,
                       0.05,
                       0.06,
                       0.03,
                       0.04,
                       0.04,
                       0.02,
                       0.04,
                       0.01]])

    k0 = k_fit[0] / k_fit[2]
    k0_error = k0 * (k_err[0] / k_fit[0] + k_err[2] / k_fit[2])

    ks = k_fit[1] / k_fit[2]
    ks_error = ks * (k_err[1] / k_fit[1] + k_err[2] / k_fit[2])

    plt.errorbar(obs_day, k0, yerr=k0_error, fmt='kx', label='k0')
    plt.errorbar(obs_day, ks, yerr=ks_error, fmt='ks', label='ks')


if (plot_knorm_all):
    fitval = fitval90

    times = np.array(range(t_start - 53000, t_end - 53000))

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "r-", label=r'i=90$^\circ$')
    plt.plot(times, k0, "r--")

    # Values from Column 2
    fitval = fitval91

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "r-", label=r'i=91.3$^\circ$')
    plt.plot(times, k0, "r--")

    # Values from Column 3
    fitval = fitval88

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "r-", label=r'i=88.7$^\circ$')
    plt.plot(times, k0, "r--")

    plt.xlabel('MJD-53000')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.show()

if (plot_knorm_VIS_inc):

    VIS0 = np.array([-79, 100]) * u.km / u.s

    fitval = fitval91

    times = np.array(range(t_start - 53000, t_end - 53000))

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "r-", label='VIS,y = 100 km/s')
    plt.plot(times, k0, "r--")

    # Rotate into parallel and perpendicular axes
    VIS = RotateVector(VIS0, -fitval['PsiAR'])
    #print('VIS after rotation: ', VIS)
    VIS[1] -= 50 * u.km / u.s  # Add to parallel axis
    #print('VIS increased parallel: ', VIS)
    # Rotate back into x and y
    fitval['VIS'] = -RotateVector(VIS, fitval['PsiAR'])

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "b-", label='VIS,par += 50 km/s')
    plt.plot(times, k0, "b--")

    # Rotate into parallel and perpendicular axes
    VIS = RotateVector(VIS0, -fitval['PsiAR'])
    VIS[0] += 50 * u.km / u.s  # Add to perpendicular axis
    #print('VIS increased perpendicular: ', VIS)
    # Rotate back into x and y
    fitval['VIS'] = -RotateVector(VIS, fitval['PsiAR'])

    #print('VIS changed parallel: ', VIS)

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "g-", label='VIS,perp += 50 km/s')
    plt.plot(times, k0, "g--")

    plt.xlabel('MJD-53000')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.show()

if (plot_knorm_VIS_dec):
    VIS0 = np.array([-79, 100]) * u.km / u.s

    fitval = fitval91

    times = np.array(range(t_start - 53000, t_end - 53000))

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "r-", label='Best Fit')
    plt.plot(times, k0, "r--")

    # Rotate into parallel and perpendicular axes
    VIS = RotateVector(VIS0, -fitval['PsiAR'])
    #print('VIS after rotation: ', VIS)
    VIS[1] = VIS[1] / 2  # Decrease parallel axis
    #print('VIS decreased parallel: ', VIS)
    # Rotate back into x and y
    fitval['VIS'] = -RotateVector(VIS, fitval['PsiAR'])

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "b-", label='VIS,par = VIS,par/2')
    plt.plot(
        times,
        k0,
        "b--from astropy.coordinates import EarthLocation, SkyOffsetFrame")

    # Rotate into parallel and perpendicular axes
    VIS = RotateVector(VIS0, -fitval['PsiAR'])
    VIS[1] = 0 * u.km / u.s  # Decrease parallel axis
    #print('VIS decreased parallel: ', VIS)
    # Rotate back into x and y
    fitval['VIS'] = -RotateVector(VIS, fitval['PsiAR'])

    [ks, k0] = k_norm(t_start, t_end, t_nsteps, fitval, psr, psr_m, bm)

    plt.plot(times, ks, "g-", label='VIS,par = 0 km/s')
    plt.plot(times, k0, "g--")

    plt.xlabel('MJD-53000')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.show()
