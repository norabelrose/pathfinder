from astropy.time import Time
from astropy.units import Unit, quantity_input
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from astropy.units.quantity import Quantity
from poliastro.bodies import Body, Earth, Mars, Sun
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from typing import Optional, Union

import numpy as np


Transfer = namedtuple('Transfer', ['launch_date', 'time_of_flight', 'maneuver'])

@quantity_input(max_tof='time')
@dataclass
class ImpulsiveTransfer:
    source: Body = Earth
    target: Union[Body, Orbit] = Mars

    # The earliest feasible launch date. Defaults to the current time.
    earliest_launch: Time = field(default_factory=lambda: Time(datetime.utcnow(), format='datetime', scale='tdb'))
    latest_arrival: Optional[Time] = None

    # Maximum time-of-flight allowed
    max_tof: Quantity = np.inf * Unit('d')

    def __post_init__(self):
        # Sanity check
        assert self.latest_arrival is None or self.latest_arrival > self.earliest_launch
        assert self.source is not self.target
        assert self.max_tof > 0.0

    # Returns a tuple of the form (launch date, maneuver), where the Maneuver object contains information about
    # the initial and final impulses.
    def compute(self, *, max_iter: int = 100, tof_weight: float = 1e-2, verbose: bool = False) -> list:
        import scipy.optimize as optim

        # The synodic period is a very useful heuristic for dividing up the search space into basins
        # that attract toward local optima. To compute the (approximate) synodic period, we take the
        # reciprocals of the periods to get the "frequencies" of the orbits in 1 / days, compute the
        # absolute difference, then convert back to a period
        orbit1 = _get_orbit(Sun, self.source, self.earliest_launch)
        orbit2 = _get_orbit(Sun, self.target, self.earliest_launch) if isinstance(self.target, Body) else self.target
        period1, period2 = orbit1.period.to('d'), orbit2.period.to('d')
        synodic_period = np.asarray(1 / abs(1 / period1 - 1 / period2))
        
        # We do the optimization in Julian days
        epoch = self.earliest_launch              # All times are measured as offsets from earliest_launch
        if not self.latest_arrival:
            # Narrow the search to the current synodic period
            t_max = synodic_period
            if verbose:
                iso_str = Time(epoch + t_max, format='jd').to_value('iso')
                print(f"Using {iso_str} (one synodic period from `earliest_launch`) as the `latest_arrival` date")
        else:
            t_max = self.latest_arrival.jd - epoch.jd

        # Constants
        days = Unit('d')
        eps = 1e-2
        tof_weight = Quantity(tof_weight, Unit('km/s') / days)

        def get_maneuver(t_launch: float, tof: float):
            t_launch = Quantity(t_launch, days).squeeze()
            tof = Quantity(tof, days).squeeze()

            source_orbit = orbit1.propagate(t_launch)
            target_orbit = orbit2.propagate(t_launch + tof)
            return Maneuver.lambert(source_orbit, target_orbit)

        # Find all of the feasible local minima a deterministic basin hopping algorithm
        def loss(t_launch: float, tof: float):
            maneuver = get_maneuver(t_launch, tof)
            loss = maneuver.get_total_cost() + (tof_weight * np.asarray(tof)) * days
            return loss.value
        
        # Optimization state
        local_minima = []
        tof_init = min(synodic_period, t_max) / 2
        tof_max = self.max_tof.to_value('d')
        t0_init = 0.0
        total_fev = 2   # We directly performed two fevals to get the initial slope

        # Jointly optimize t0 and tof to find the next local minimum, adding it to the stack.
        # Returns TRUE if the outer loop should stop.
        def next_local_minimum() -> bool:
            nonlocal local_minima, t0_init, tof_init, total_fev
            result = optim.minimize(
                lambda x: loss(*x),
                x0=(t0_init, tof_init),
                bounds=(
                    (t0_init, t_max),                     # Don't allow "backtracking" to launch dates we've already explored
                    (eps, min(tof_max, t_max - t0_init))  # ToF should be greater than zero for numerical stability
                )
            )
            total_fev += result.nfev
            
            # The inner loop diverged, so let's bail
            if not result.success:
                return True
            
            t0, tof = result.x
            arrival = t0 + tof

            # The next local minimum is outside of our launch window, so let's bail
            if arrival > t_max:
                return True
            
            t0_init, tof_init = t0, tof
            local_minima.append((t0, tof))

            # If this local minimum is right up against our launch window upper bound,
            # we SHOULD return it, and then bail
            if np.isclose(arrival, t_max, atol=eps):
                return True

        # Jump to the edge of the next basin in the cost landscape by setting t0_init to epsilon
        # greater than the next local *maximum*, while keeping ToF constant
        def next_t0_peak() -> bool:
            nonlocal t0_init, total_fev
            result = optim.minimize_scalar(
                lambda t0: -loss(t0, tof_init),
                bracket=(t0_init, t0_init + synodic_period / 2),
                tol=eps
            )
            total_fev += result.nfev
            t0_init = 1.0 + result.x
            
            if not result.success or result.x >= t_max - eps:
                return True

        # First we need to figure out whether we're starting out at a point of positive or negative
        # slope in the cost landscape along the launch date axis. If positive, we need to find the
        # first *local maximum* along this axis, then descend the gradient forward in time to get
        # our first feasible local minimum. For this we use a finite differences approximation.
        slope_init = (loss(t0_init + eps, tof_init) - loss(t0_init, tof_init)) / eps
        if slope_init > 0.0:
            next_t0_peak()
        
        # The core outer loop
        for _ in range(max_iter):
            if next_local_minimum():
                break

            if next_t0_peak():
                break
        
        print(f"Total function evals: {total_fev}")
        return [Transfer(epoch + t0, tof * days, get_maneuver(t0, tof)) for t0, tof in local_minima]


# Convenience method
def _get_orbit(attractor: Body, planet: Body, epoch: Time) -> Orbit:
     return Orbit.from_ephem(attractor, Ephem.from_body(planet, epoch), epoch)
