import numpy as np
import pandas as pd
from pathlib import Path

def load_annotated(dir: str="experiment/data/annotated/") -> pd.DataFrame:
    dats = []
    for f in Path(dir).iterdir():
        if f.suffix != ".csv":
            continue
        dats.append(
            pd.read_csv(f, parse_dates=["last_update"])
        )
    dat = pd.concat(dats)
    dat = dat[['collected_at', 'sensor_name', 'last_update', 'trigger', 'annotation']]
    return dat

def prepare_data(dat: pd.DataFrame) -> pd.DataFrame:
    dat = dat.copy()
    dat = dat[~pd.isna(dat['annotation'])]
    dat = dat[dat['annotation'].str.endswith("r")]
    dat = dat[['last_update', 'sensor_name', 'annotation']]
    dat['annotation'] = dat['annotation'].apply(lambda x: int(x[0]))
    return dat

dat = load_annotated()
dat = prepare_data(dat)

def cyclical_time(dat: pd.DataFrame, target_col: str):

    def apply_cyclical_time(df, col, max_t):
        ang = 2 * np.pi * df[col] / max_t
        df[col + "_sin"] = np.sin(ang)
        df[col + "_cos"] = np.cos(ang)

    def seconds_of_day(hour, minute, second):
        return hour * 3600 + minute * 60 + second

    dat_dt = dat[target_col].dt
    dat['year'] = dat_dt.year
    dat['dayofyear'] = dat_dt.dayofyear - 1
    dat['secondofday'] = seconds_of_day(dat_dt.hour, dat_dt.minute, dat_dt.second) - 1
    dat['weekday'] = dat_dt.weekday

    apply_cyclical_time(dat, 'weekday', 7)
    apply_cyclical_time(dat, 'dayofyear', 365)
    apply_cyclical_time(dat, 'secondofday', 86400)

    return dat

from src.caching import pickle_load_from_file
g = pickle_load_from_file("preprocessed/graph/exp-full-pruned-prob.pkl")

def apply_sensor_id(dat: pd.DataFrame, sensor_name_to_id: dict[str, int]):
    dat['sensor_id'] = dat['sensor_name'].apply(lambda x: sensor_name_to_id[x.upper()])
    return dat
dat = apply_sensor_id(dat, g['sensor_name_to_id'])
dat = cyclical_time(dat, "last_update")

# fix annotation
dat['annotation'].value_counts()
dat['annotation'] -= 1
dat.loc[dat['annotation'] == 3, 'annotation'] -= 1

X_names = ["year", "weekday_sin", "weekday_cos", "dayofyear_sin",
           "dayofyear_cos", "secondofday_sin", "secondofday_cos", "sensor_id"]
y_name = "annotation"