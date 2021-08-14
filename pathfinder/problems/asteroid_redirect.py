import numpy as np
import pygmo as pg
import pykep as pk


MASS_16_PSYCHE = 2.2923752e10   # kilograms


class AsteroidRedirect(pg.problem.base):
    def __init__(self, isp: float, thrust: float):
        super().__init__()

        # For a fixed mission start time, this is a continuous optimal control problem which can
        # be solved indirectly using Pontryagin's maximum principle.

        # spacecraft
        self.sc = pk.sims_flanagan.spacecraft(MASS_16_PSYCHE, thrust, isp)

        # indirect leg
        self.leg = pk.pontryagin.leg(
            sc=self.sc, mu=pk.MU_SUN, freemass=True, freetime=True, alpha=alpha, bound=bound
        )

        # integration parameters
        if all([(isinstance(par, float) or isinstance(par, int)) for par in [atol, rtol]]):
            self.atol = float(atol)
            self.rtol = float(rtol)
        else:
            raise TypeError(
                "Both atol and rtol must be an instance of either float or int.")

    def _objfun_impl(self):
        pass

    def fitness(self, z):
        # times
        t0 = self.t0
        tf = pk.epoch(t0.mjd2000 + z[0])

        # intial costates
        l0 = np.asarray(z[1:])

        # arrival conditions
        rf, vf = self.pf.eph(tf)

        # departure state
        x0 = pk.sims_flanagan.sc_state(self.x0[0:3], self.x0[3:6], self.x0[6])

        # arrival state (mass will be ignored)
        xf = pk.sims_flanagan.sc_state(rf, vf, self.sc.mass / 10)

        # set leg
        self.leg.set(t0, x0, l0, tf, xf)

        # equality constraints
        ceq = self.leg.mismatch_constraints(atol=self.atol, rtol=self.rtol)

        obj = self.leg.trajectory[-1, -1] * self.leg._dynamics.c2 * 1000

        return np.hstack(([obj], ceq))

    def get_bounds(self):
        lb = [self.tof[0]] + [-1e2] * 7
        ub = [self.tof[1]] + [1e2] * 7
        return (lb, ub)

    def _plot_traj(self, z, axes, units=pk.AU):
        """Plots spacecraft trajectory.

        Args:
            - z (``tuple``, ``list``, ``numpy.ndarray``): Decision chromosome.
            - axes (``matplotlib.axes._subplots.Axes3DSubplot``): 3D axes to use for the plot
            - units (``float``, ``int``): Length unit by which to normalise data.

        Examples:
            >>> prob.extract(pykep.trajopt.indirect_pt2or).plot_traj(pop.champion_x)
        """

        # states
        x0 = self.x0

        # times
        t0 = self.t0
        tf = pk.epoch(t0.mjd2000 + z[0])

        # Computes the osculating Keplerian elements at start
        elem0 = list(pk.ic2par(x0[0:3], x0[3:6], self.leg.mu))

        # Converts the eccentric anomaly into eccentric anomaly
        elem0[5] = elem0[5] - elem0[1] * np.sin(elem0[5])

        # Creates a virtual keplerian planet with the said elements
        kep0 = pk.planet.keplerian(t0, elem0)

        # Plots the departure and arrival osculating orbits
        pk.orbit_plots.plot_planet(
            kep0, t0, units=units, color=(0.8, 0.8, 0.8), axes=axes)
        pk.orbit_plots.plot_planet(
            self.pf, tf, units=units, color=(0.8, 0.8, 0.8), axes=axes)

    def _pretty(self, z):
        print("\nPlanet to orbit transfer, alpha is: ",  self._alpha)
        print("\nFrom (cartesian): " + str(list(self.x0)))
        print("Launch epoch: {!r} MJD2000, a.k.a. {!r}".format(
            self.t0.mjd2000, self.t0))
        print("\nTo (planet): " + self.pf.name)
        print("Time of flight (days): {!r} ".format(z[0]))
    