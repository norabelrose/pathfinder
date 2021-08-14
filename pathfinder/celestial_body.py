from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.units import Quantity
from astroquery.jplsbdb import SBDB
from dataclasses import dataclass


class CelestialBody:
    def __init__(self, name: str):
        pass


@dataclass
class SmallBody:
    name: str

    # The six Keplerian elements needed to estimate the position of the
    # small body at any given point in time
    semimajor_axis: Quantity
    eccentricity: Quantity

    inclination: Quantity
    long_of_asc_node: Quantity
    arg_of_periapsis: Quantity

    true_anomaly: Quantity
    epoch: Quantity

    @staticmethod
    def with_name(name: str) -> 'SmallBody':
        results = SBDB.query(name)  # type: ignore
        orbit = results['orbit']
        elements = orbit['elements']

        return SmallBody(
            name=name,
            semimajor_axis=elements['a'],
            eccentricity=elements['e'],
            inclination=elements['i'],
            long_of_asc_node=elements['n'],
            arg_of_periapsis=elements['w'],
            true_anomaly=elements[''],
            epoch=orbit['epoch']
        )

    def compute_position(self, t: Time):
        pass
