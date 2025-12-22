# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Tuple, List, Dict, Optional
import os
import random
import requests

import pandas as pd
import bittensor as bt
import numpy as np
import torch
import h3

from zeus.data.weatherxm.station import Station
from zeus.utils.time import get_today, to_timestamp
from zeus.data.base.loader import BaseDataLoader
from zeus.data.base.predictor import BasePredictor
from zeus.data.weatherxm.sample import WeatherXMSample
from zeus.validator.constants import MechanismType
from zeus.validator.constants import (
    WEATHER_XM_URL,
    WEATHER_XM_DATA_VARS,
    WEATHERXM_START_SAMPLE_STD,
    WEATHERXM_MIN_DATA_QUALITY,
    WEATHERXM_START_OFFSET_RANGE,
    HOURS_PREDICT_RANGE,
    WEATHERXM_CELL_RESOLUTION,
)
import os
import json


class WeatherXMLoader(BaseDataLoader, BasePredictor):

    DATA_INTERVAL = pd.Timedelta(hours=1)
    mechanism = MechanismType.WEATHER_XM
    sample_cls = WeatherXMSample

    def __init__(
        self,
        base_url: str = WEATHER_XM_URL,
        start_sample_std: float = WEATHERXM_START_SAMPLE_STD,
    ):
        super().__init__(
            data_vars=WEATHER_XM_DATA_VARS,
            predict_sample_range=HOURS_PREDICT_RANGE,
            start_offset_range=WEATHERXM_START_OFFSET_RANGE,
        )

        self.start_sample_std = start_sample_std

        self.base_url = base_url
        self.api_key = os.getenv(
            "WEATHERXM_API_KEY", "1e68095e-bab0-4ef1-b2ac-2b961daf82fd"
        )

        try:
            self.stations = self.load_all_stations()
            assert len(self.stations) > 100
        except Exception as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you setup your WeatherXM API Key?")

    def _get_headers(self) -> Dict[str, str]:
        """
        Returns the headers for the WeatherXM API.
        """
        return {
             "Accept": "application/json",
            "X-API-KEY": self.api_key
        }


    def load_all_stations(self) -> Dict[str, Station]:
        """
        Returns all the stations from the WeatherXM API.
        Uses a simple JSON file cache to avoid unnecessary HTTP requests.
        Returns:
            Dict[str, Station]: The dictionary of stations keyed by station id.
        """
        cache_file = "stations_cache.json"

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                stations: Dict[str, Station] = {}
                for dp in cached_data:
                    try:
                        station = Station(
                            id=dp["id"],
                            latitude=dp["lat"],
                            longitude=dp["lon"],
                            elevation=dp["elevation"],
                            lastDayQod=dp["lastDayQod"],
                        )
                        if None not in [
                            station.id,
                            station.latitude,
                            station.longitude,
                            station.elevation,
                            station.lastDayQod,
                        ]:
                            stations[station.id] = station
                    except Exception:
                        # Ignore malformed entries in cache
                        pass

                bt.logging.info(
                    f"[WeatherXMLoader] Loaded {len(stations)} stations from cache"
                )
                return stations
            except Exception as e:
                bt.logging.warning(
                    f"[WeatherXMLoader] Failed to load stations cache: {e}. "
                    f"Falling back to API request."
                )

        # If cache does not exist or failed, call the API
        response = requests.get(
            self.base_url + "/stations", headers=self._get_headers()
        )
        response.raise_for_status()

        data = response.json()

        stations: Dict[str, Station] = {}
        for dp in data:
            try:
                station = Station(
                    id=dp["id"],
                    latitude=dp["lat"],
                    longitude=dp["lon"],
                    elevation=dp["elevation"],
                    lastDayQod=dp["lastDayQod"],
                )
                if None not in [
                    station.id,
                    station.latitude,
                    station.longitude,
                    station.elevation,
                    station.lastDayQod,
                ]:
                    stations[station.id] = station
            except Exception:
                # Ignore malformed entries from API
                pass

        # Save raw API response to cache file
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            bt.logging.warning(f"[WeatherXMLoader] Failed to write stations cache: {e}")

        bt.logging.info(
            f"[WeatherXMLoader] Loaded {len(stations)} stations around the globe"
        )
        return stations

    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample a time range for a sample.
        Returns:
            pd.Timestamp: The start timestamp.
            pd.Timestamp: The end timestamp.
            int: The number of predict hours.
        """
        num_predict_hours = np.random.randint(*self.predict_sample_range)
        start_offset = (
            min(
                self.start_offset_range[1],
                np.abs(int(np.random.normal(0, self.start_sample_std))),
            )
            + self.start_offset_range[0]
        )  # always into the future

        start_timestamp = get_today("h") + pd.Timedelta(hours=start_offset)
        end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours - 1)

        return start_timestamp, end_timestamp, num_predict_hours

    def get_sample(self) -> WeatherXMSample:
        """
        Get a current sample.

        Returns:
        - sample (WeatherXMSample): The sample containing the coordinates and dates. Output data is not yet known.
        """
        station = random.choice(list(self.stations.values()))
        start_time, end_time, predict_hours = self.sample_time_range()

        return WeatherXMSample(
            lat=station.latitude,
            lon=station.longitude,
            elevation=station.elevation,
            station_id=station.id,
            variable=self.sample_variable(),
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            hours_to_predict=predict_hours,
        )

    def get_last_available(self) -> pd.Timestamp:
        """
        Returns the last available timestamp, which is the last hour in the previous day.
        Returns:
            pd.Timestamp: The last available timestamp.
        """
        return get_today(floor="d") - pd.Timedelta(hours=1)

    def get_output(self, sample: WeatherXMSample) -> Optional[torch.Tensor]:
        """
        Get the output data for a sample.
        Returns:
            Optional[torch.Tensor]: The output data, or None if the sample is not available.
        """
        start = to_timestamp(sample.start_timestamp)
        end = to_timestamp(sample.end_timestamp)
        if end >= self.get_last_available():
            return None

        def get_data(day: pd.Timestamp) -> List[Tuple[pd.Timestamp, float]]:
            """
            WeatherXM returns history in GMT+0 by default
            """
            response = requests.get(
                self.base_url + f"/stations/{sample.station_id}/history",
                headers=self._get_headers(),
                params={"date": day.strftime("%Y-%m-%d")},
            ).json()

            if len(response) == 1:  # likely an error message
                return []

            result = []
            for dp in response:
                try:
                    if (
                        dp["health"]["data_quality"]["score"]
                        < WEATHERXM_MIN_DATA_QUALITY
                    ):
                        continue
                    observation = dp["observation"]
                    value = observation[sample.variable]
                    if value is not None:
                        result.append(
                            (
                                pd.Timestamp(observation["timestamp"]).replace(
                                    tzinfo=None
                                ),
                                value,
                            )
                        )
                except:
                    pass  # faulty datapoints are ignored

            return result

        raw_data = []
        # only request twice if start and end on different day
        for day in set([start.floor("d"), end.floor("d")]):
            raw_data.extend(get_data(day))

        if not raw_data:
            return

        output = []
        desired_hour = start
        best_distance = np.inf
        # find closest datapoint for each desired hour
        for dp_time, dp_value in raw_data:
            distance = abs((desired_hour - dp_time).total_seconds())
            if distance < best_distance:
                cur_best_value = dp_value
                best_distance = distance
            # we were previously as close as possible, use that datapoint
            else:
                output.append(cur_best_value)
                best_distance = np.inf
                desired_hour = desired_hour + self.DATA_INTERVAL
                if desired_hour > end:
                    return torch.Tensor(output)

        # in case last datapoint is very close to end timestamp
        output.append(cur_best_value)
        return torch.Tensor(output)

    def get_forecast(self, sample: WeatherXMSample) -> Optional[torch.Tensor]:        
        """
        Get the forecast for a sample using the WeatherXM V1 API.
        Returns:
            torch.Tensor: The forecast.
        """
        start_time = to_timestamp(sample.start_timestamp)
        end_time = to_timestamp(sample.end_timestamp)
        h3cell = h3.latlng_to_cell(
            sample.lat, sample.lon, res=WEATHERXM_CELL_RESOLUTION
        )

        api_response = requests.get(
            self.base_url + f"/cells/{h3cell}/forecast/wxmv1",
            params={
                "include": "hourly",
                "from": start_time.strftime("%Y-%m-%d"),
                "to": end_time.strftime("%Y-%m-%d"),
            },
            headers=self._get_headers()
        )
        if not api_response.ok:
            return
        cell_pred = api_response.json()

        output = np.concat(
            [
                np.asarray([hour_data[sample.variable] for hour_data in day["hourly"]])
                for day in cell_pred
            ],
        )[start_time.hour : -(23 - end_time.hour)]

        return torch.Tensor(output)
