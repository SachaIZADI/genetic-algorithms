import pandas as pd
import geopandas as gpd
from pathlib import Path


def get_tsp_data():

    PATH = Path(__file__).parent

    latitude_longitude_coordinates = pd.read_fwf(PATH / "uscap_ll.txt", names=["latitude", "longitude"])
    names = pd.read_fwf(PATH / "uscap_name.txt", names=["city"])
    x_y_coordinates = pd.read_fwf(PATH / "uscap_xy.txt", names=["x", "y"])

    tsp_data = pd.concat([names, latitude_longitude_coordinates, x_y_coordinates], axis=1)

    tsp_data = tsp_data[~tsp_data["city"].isin([
        "Honolulu, Hawaii",
        "Juneau, Alaska",
    ])].reset_index(drop=True)

    return tsp_data


def get_usa_map():

    PATH = Path(__file__).parent
    usa_states = gpd.read_file(PATH / 'usa_states/usa-states-census-2014.shp')

    return usa_states