from astropy.time import Time
from astropy.units import Quantity
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Objective:
    # These attribute the earliest time that burns may occur and the latest time before which
    # the objective must be accomplished, respectively.
    min_time: Time = field(default_factory=lambda: Time.now())
    max_time: Optional[Time] = None

    # Maximum amount of delta v that can be expended before a path is considered to be infeasible
    delta_v_budget: Optional[Quantity] = None

    def __post_init__(self):
        # Sanity checks
        assert not self.max_time or self.max_time > self.min_time
        if self.delta_v_budget is not None:
            assert self.delta_v_budget > 0.0
            self.delta_v_budget = self.delta_v_budget.to('m/s')  # Do all computations in meters per second
