from astropy.units import Quantity
from astroquery.jplhorizons import Horizons
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import numpy as np

from .constants import planets


@dataclass
class KeplerianElements:
    # The six Keplerian elements needed to estimate the position of the
    # small body at any given point in time
    semimajor_axis: Quantity
    eccentricity: Quantity

    inclination: Quantity
    long_of_asc_node: Quantity
    arg_of_periapsis: Quantity

    true_anomaly: Quantity

    @staticmethod
    @lru_cache()
    def get_body(name: str, time: Any = None) -> 'KeplerianElements':
        try:
            number = planets.index(name.lower()) + 1
        except ValueError:
            name_type = 'name'
        else:
            name = str(number) + '99'
            name_type = 'id'
        
        elem = Horizons(name, epochs=time, id_type=name_type).elements()    # type: ignore
        return KeplerianElements(
            elem['a'],
            elem['e'],
            elem['incl'],
            elem['Omega'],
            elem['w'],
            elem['nu']
        )

    @staticmethod
    def from_newtonian(r: Quantity, v: Quantity, mu: Quantity):
        # Sanity check, we need to be in 3D space
        assert r.shape[-1] == v.shape[-1] == 3

        # Compute angular momentum
        h = np.cross(r, v)

        # Compute longitude of the ascending node
        n = np.cross(np.array([0.0, 0.0, 1.0]), h)

        r_norm, v_sq = np.linalg.norm(r), np.linalg.norm(v) ** 2
        rv_prod = np.sum(r * v, axis=-1)

        e_vec = np.cross(v, h) / mu - r / r_norm
        e = np.linalg.norm(e_vec)

        energy = 0.5 * v_sq - mu / r_norm
        h_norm, n_norm = np.linalg.norm(h), np.linalg.norm(n)
        a = -mu / (2 * energy) if np.abs(e - 1.0) > 1e-4 else np.inf
        i = np.arccos(h[2] / h_norm)

        omega = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            omega = 2 * np.pi - omega

        argp = np.arccos(np.sum(n * e_vec, axis=-1) / (n_norm * e))
        if e[2] < 0:
            argp = 2 * np.pi - argp

        nu = np.arccos(np.sum(e_vec * r, axis=-1) / (e * r_norm))
        if rv_prod < 0:
            nu = 2 * np.pi - nu
        
        return KeplerianElements(a, e, i, omega, argp, nu)
    
    def get_eccentric_anomaly(self) -> Quantity:
        numer = np.sqrt(1 - self.eccentricity ** 2) * np.sin(self.true_anomaly)
        denom = self.eccentricity + np.cos(self.true_anomaly)
        return np.arctan(numer / denom) * self.true_anomaly.unit
    
    def get_orbital_radius(self, E = None) -> Quantity:
        E = self.get_eccentric_anomaly() if E is None else E
        return self.semimajor_axis * (1 - self.eccentricity * np.cos(E))
