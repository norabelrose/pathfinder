import pykep as pk
from datetime import date, datetime
from pykep.trajopt import pl2pl_N_impulses


class EarthToAsteroid(pl2pl_N_impulses):
    """
    This problem works by manipulating the starting epoch t0, the transfer time T the final mass mf and the controls 
    The decision vector is::

        z = [t0, T, mf, Vxi, Vyi, Vzi, Vxf, Vyf, Vzf, controls]
    """

    def __init__(self,
                 target=pk.jpl_lp('mars'),
                 N_max=2,
                 tof=[20., 400.],
                 vinf=[0., 4.],
                 phase_free=True,
                 multi_objective=False,
                 t0=None):
        # 2) Number of impulses must be at least 2
        if N_max < 2:
            raise ValueError('Number of impulses N is less than 2')
        
        start_epoch = pk.epoch_from_string(datetime.now().isoformat().replace('T', ' '))
        if phase_free:
            if t0 is not None:
                raise ValueError('When phase_free is True no t0 can be specified')
        else:
            if t0 is None:
                end_epoch = pk.epoch(start_epoch.jd + 1000)
                t0 = [start_epoch, end_epoch]
            
            if (type(t0[0]) != type(start_epoch)):
                t0[0] = pk.epoch(t0[0])
            if (type(t0[1]) != type(start_epoch)):
                t0[1] = pk.epoch(t0[1])
        
        # Basic assumptions of the problem
        start = pk.jpl_lp('earth')
        self.obj_dim = multi_objective + 1

        # We then define all class data members
        self.start = start
        self.target = target
        self.N_max = N_max
        self.phase_free = phase_free
        self.multi_objective = multi_objective
        self.vinf = [s * 1000 for s in vinf]

        self.__common_mu = start.mu_central_body

        # And we compute the bounds
        if phase_free:
            self._lb = [0, tof[0]] + [1e-3, 0.0, 0.0, vinf[0] * 1000] * (N_max - 2) + [1e-3] + [0]
            self._ub = [2 * start.compute_period(start_epoch) * pk.SEC2DAY, tof[1]] + [1.0-1e-3, 1.0, 1.0, vinf[
                1] * 1000] * (N_max - 2) + [1.0-1e-3] + [2 * target.compute_period(start_epoch) * pk.SEC2DAY]
        else:
            self._lb = [t0[0].mjd2000, tof[0]] + \
                [1e-3, 0.0, 0.0, vinf[0] * 1000] * (N_max - 2) + [1e-3]
            self._ub = [t0[1].mjd2000, tof[1]] + \
                [1.0-1e-3, 1.0, 1.0, vinf[1] * 1000] * (N_max - 2) + [1.0-1e-3]

    def get_nobj(self):
        return self.obj_dim

    def get_bounds(self):
        return (self._lb, self._ub)

    def fitness(self, x):
        # 1 -  we 'decode' the chromosome into the various deep space
        # maneuvers times (days) in the list T
        T = list([0] * (self.N_max - 1))

        for i in range(len(T)):
            T[i] = pk.log(x[2 + 4 * i])
        total = sum(T)
        T = [x[1] * time / total for time in T]

        # 2 - We compute the starting and ending position
        r_start, v_start = self.start.eph(pk.epoch(x[0]))
        if self.phase_free:
            r_target, v_target = self.target.eph(pk.epoch(x[-1]))
        else:
            r_target, v_target = self.target.eph(pk.epoch(x[0] + x[1]))

        # 3 - We loop across inner impulses
        rsc = r_start
        vsc = v_start
        for i, time in enumerate(T[:-1]):
            theta = 2 * pk.pi * x[3 + 4 * i]
            phi = pk.acos(2 * x[4 + 4 * i] - 1) - pk.pi / 2

            Vinfx = x[5 + 4 * i] * pk.cos(phi) * pk.cos(theta)
            Vinfy = x[5 + 4 * i] * pk.cos(phi) * pk.sin(theta)
            Vinfz = x[5 + 4 * i] * pk.sin(phi)

            # We apply the (i+1)-th impulse
            vsc = [a + b for a, b in zip(vsc, [Vinfx, Vinfy, Vinfz])]
            rsc, vsc = pk.propagate_lagrangian(rsc, vsc, time * pk.DAY2SEC, self.__common_mu)
        cw = (pk.ic2par(rsc, vsc, self.start.mu_central_body)[2] > pk.pi / 2)

        # We now compute the remaining two final impulses
        # Lambert arc to reach seq[1]
        dt = T[-1] * pk.DAY2SEC
        l = pk.lambert_problem(rsc, r_target, dt, self.__common_mu, cw, False)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        DV1 = pk.norm([a - b for a, b in zip(v_beg_l, vsc)])
        DV2 = pk.norm([a - b for a, b in zip(v_end_l, v_target)])

        DV_others = sum(x[5::4])
        if self.obj_dim == 1:
            return (DV1 + DV2 + DV_others,)
        else:
            return (DV1 + DV2 + DV_others, x[1])

    def plot(self, x, axes=None):
        """
        ax = prob.plot_trajectory(x, axes=None)

        - x: encoded trajectory
        - axes: matplotlib axis where to plot. If None figure and axis will be created
        - [out] ax: matplotlib axis where to plot

        Plots the trajectory represented by a decision vector x on the 3d axis ax

        Example::

          ax = prob.plot(x)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

        if axes is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axes = fig.gca(projection='3d')

        axes.scatter(0, 0, 0, color='y')

        # 1 -  we 'decode' the chromosome recording the various deep space
        # maneuvers timing (days) in the list T
        T = list([0] * (self.N_max - 1))

        for i in range(len(T)):
            T[i] = pk.log(x[2 + 4 * i])
        total = sum(T)
        T = [x[1] * time / total for time in T]

        # 2 - We compute the starting and ending position
        r_start, v_start = self.start.eph(pk.epoch(x[0]))
        if self.phase_free:
            r_target, v_target = self.target.eph(pk.epoch(x[-1]))
        else:
            r_target, v_target = self.target.eph(pk.epoch(x[0] + x[1]))
        plot_planet(self.start, t0=pk.epoch(x[0]), color=(0.8, 0.6, 0.8), legend=True, units=pk.AU, axes=axes, s=0)
        plot_planet(self.target, t0=pk.epoch(x[0] + x[1]), color=(0.8, 0.6, 0.8), legend=True, units=pk.AU, axes=axes, s=0)

        DV_list = x[5::4]
        maxDV = max(DV_list)
        DV_list = [s / maxDV * 30 for s in DV_list]
        colors = ['b', 'r'] * (len(DV_list) + 1)

        # 3 - We loop across inner impulses
        rsc = r_start
        vsc = v_start
        for i, time in enumerate(T[:-1]):
            theta = 2 * pk.pi * x[3 + 4 * i]
            phi = pk.acos(2 * x[4 + 4 * i] - 1) - pk.pi / 2

            Vinfx = x[5 + 4 * i] * pk.cos(phi) * pk.cos(theta)
            Vinfy = x[5 + 4 * i] * pk.cos(phi) * pk.sin(theta)
            Vinfz = x[5 + 4 * i] * pk.sin(phi)

            # We apply the (i+1)-th impulse
            vsc = [a + b for a, b in zip(vsc, [Vinfx, Vinfy, Vinfz])]
            axes.scatter(rsc[0] / pk.AU, rsc[1] / pk.AU, rsc[2] /
                         pk.AU, color='k', s=DV_list[i])
            plot_kepler(rsc, vsc, T[i] * pk.DAY2SEC, self.__common_mu,
                        N=200, color=colors[i], units=pk.AU, axes=axes)
            rsc, vsc = pk.propagate_lagrangian(
                rsc, vsc, T[i] * pk.DAY2SEC, self.__common_mu)

        cw = (pk.ic2par(rsc, vsc, self.start.mu_central_body)[2] > pk.pi / 2)
        # We now compute the remaining two final impulses
        # Lambert arc to reach seq[1]
        dt = T[-1] * pk.DAY2SEC
        l = pk.lambert_problem(rsc, r_target, dt, self.__common_mu, cw, False)
        plot_lambert(l, sol=0, color=colors[
                     i + 1], legend=False, units=pk.AU, axes=axes, N=200)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        DV1 = pk.norm([a - b for a, b in zip(v_beg_l, vsc)])
        DV2 = pk.norm([a - b for a, b in zip(v_end_l, v_target)])

        axes.scatter(rsc[0] / pk.AU, rsc[1] / pk.AU, rsc[2] / pk.AU,
                     color='k', s=min(DV1 / maxDV * 30, 40))
        axes.scatter(r_target[0] / pk.AU, r_target[1] / pk.AU,
                     r_target[2] / pk.AU, color='k', s=min(DV2 / maxDV * 30, 40))

        return axes

    def pretty(self, x):
        # 1 -  we 'decode' the chromosome recording the various deep space
        # maneuvers timing (days) in the list T
        T = list([0] * (self.N_max - 1))

        for i in range(len(T)):
            T[i] = pk.log(x[2 + 4 * i])
        total = sum(T)
        T = [x[1] * time / total for time in T]

        # 2 - We compute the starting and ending position
        r_start, v_start = self.start.eph(pk.epoch(x[0]))
        if self.phase_free:
            r_target, v_target = self.target.eph(pk.epoch(x[-1]))
        else:
            r_target, v_target = self.target.eph(pk.epoch(x[0] + x[1]))

        # 3 - We loop across inner impulses
        rsc = r_start
        vsc = v_start
        for i, time in enumerate(T[:-1]):
            theta = 2 * pk.pi * x[3 + 4 * i]
            phi = pk.acos(2 * x[4 + 4 * i] - 1) - pk.pi / 2

            Vinfx = x[5 + 4 * i] * pk.cos(phi) * pk.cos(theta)
            Vinfy = x[5 + 4 * i] * pk.cos(phi) * pk.sin(theta)
            Vinfz = x[5 + 4 * i] * pk.sin(phi)

            # We apply the (i+1)-th impulse
            vsc = [a + b for a, b in zip(vsc, [Vinfx, Vinfy, Vinfz])]
            rsc, vsc = pk.propagate_lagrangian(
                rsc, vsc, T[i] * pk.DAY2SEC, self.__common_mu)
        cw = (pk.ic2par(rsc, vsc, self.start.mu_central_body)[2] > pk.pi / 2)

        # We now compute the remaining two final impulses
        # Lambert arc to reach seq[1]
        dt = T[-1] * pk.DAY2SEC
        l = pk.lambert_problem(rsc, r_target, dt, self.__common_mu, cw, False)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        DV1 = pk.norm([a - b for a, b in zip(v_beg_l, vsc)])
        DV2 = pk.norm([a - b for a, b in zip(v_end_l, v_target)])

        DV_others = list(x[5::4])
        DV_others.extend([DV1, DV2])

        print("Total DV (m/s): ", sum(DV_others))
        print("Dvs (m/s): ", DV_others)
        print("Tofs (days): ", T)
