import os
import pandas as pd
import requests


# Defined to be exactly this number since 2012
METERS_PER_AU = 149_597_870_700

_asteroid_info: pd.DataFrame = None


def get_asteroid_info(cache_file_name: str = 'jpl-asteriod-info.csv', *, force_refresh: bool = False) -> pd.DataFrame:
    global _asteroid_info

    # We previously loaded it in this Python interpreter session
    if not force_refresh and _asteroid_info is not None:
        return _asteroid_info
    
    # Keplerian elements & their uncertainty
    kepler = ['e', 'a', 'i', 'om', 'w', 'ma']
    kepler_sigma = ['sigma_' + x for x in kepler]
    physical = ['diameter', 'diameter_sigma', 'epoch']
    
    numeric_cols = kepler + kepler_sigma + physical
    string_cols = ['full_name', 'spec_T', 'spec_B']
    col_names = ['spkid'] + numeric_cols
    
    # Check if we have it cached already
    if os.path.exists(cache_file_name):
        print(f"Loading cached asteroid info from '{cache_file_name}'")

        # By default Pandas doesn't always get this right
        dtypes = {col: 'float' for col in numeric_cols}
        dtypes |= {col: 'string' for col in string_cols}
        dtypes['spkid'] = 'int'

        _asteroid_info = pd.read_csv(cache_file_name, index_col='spkid', dtype=dtypes)
    else:
        # We need to request the data from the JPL Small Body Database
        print("Downloading asteroid info from the JPL Small Body Database...")

        response = requests.get('https://ssd-api.jpl.nasa.gov/sbdb_query.api', params={
            'sb-kind': 'a',
            
            # Get only the asteroids with a known diameter value
            'sb-cdata': r"""
            {
                "AND": ["diameter|DF"]
            }
            """,
            'fields': ','.join(col_names)
        })
        response.raise_for_status()     # Assert that status == 200

        dict_response = response.json()
        _asteroid_info = pd.DataFrame(dict_response['data'], columns=dict_response['fields']).infer_objects()

        # _asteroid_info.spkid = _asteroid_info.spkid.astype('int')
        _asteroid_info.set_index('spkid', inplace=True)
        _asteroid_info.to_csv(cache_file_name)
        print(f"Done. Cached asteroid info for future use at '{cache_file_name}'.")
    
    return _asteroid_info


MARS_APHELION = 1.666       # 249_200_000
MARS_PERIHELION = 1.382     # 206_700_000

def get_mars_crossers():
    info = get_asteroid_info()
    perihelions = info.a * (1 - info.e)
    return info.loc[perihelions <= MARS_APHELION]


def search_asteroids_by_name(search_query: str) -> pd.DataFrame:
    info = get_asteroid_info()
    return info.loc[info.full_name.str.contains(search_query, case=False)]
