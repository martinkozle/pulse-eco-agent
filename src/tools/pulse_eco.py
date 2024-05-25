import datetime

import geopy
import geopy.distance
import httpx
from langchain.tools import tool
from opencage.geocoder import OpenCageGeocode
from pulseeco import AveragePeriod, DataValueType, PulseEcoClient
from pydantic.v1 import BaseModel, Field

from src.environ import OPENCAGE_API


class GetCurrentCityAirQuality(BaseModel):
    city: str = Field(..., description="City to get the air quality stats of.")


class GetAverageCityAirQualityOnDate(BaseModel):
    city: str = Field(description="City to get the air quality stats for")
    date: datetime.date = Field(
        description="ISO date for when to get the air quality stats"
    )


class GetAverageLocationAirQualityOnDate(BaseModel):
    city: str = Field(description="City to get the air quality stats for")
    location: str = Field(
        description="Free form location query for exact location air quality"
    )
    date: datetime.date = Field(
        description="ISO date for when to get the air quality stats"
    )


@tool("get_current_city_air_quality", args_schema=GetCurrentCityAirQuality)
def get_current_city_air_quality(city: str) -> str:
    """Get the current real-time overall air quality stats of a specific input city."""
    with httpx.Client() as client:
        pulse_eco_client = PulseEcoClient(city_name=city, client=client)
        try:
            values = pulse_eco_client.overall().values
        except httpx.HTTPError:
            return f"Invalid city name `{city}`."
    values_str = ", ".join(
        f"{key}: {value}" for key, value in values if value is not None
    )
    if len(values_str) == 0:
        return "No data available for this city."
    return f"Current air quality for `{city}`: {values_str}"


@tool(
    "get_average_city_air_quality_on_date", args_schema=GetAverageCityAirQualityOnDate
)
def get_average_city_air_quality_on_date(city: str, date: datetime.date) -> str:
    """Get the day average air quality stats of a specific input city on a date."""

    if date == datetime.datetime.now(tz=datetime.UTC).date():
        return (
            "Using this tool for today's date is not supported!"
            " You can only get daily average for a finished day."
        )
    next_day = date + datetime.timedelta(days=1)
    with httpx.Client() as client:
        pulse_eco_client = PulseEcoClient(city_name=city, client=client)
        try:
            avg_data_nested = [
                pulse_eco_client.avg_data(
                    period=AveragePeriod.DAY,
                    from_=datetime.datetime(  # noqa: DTZ001
                        date.year, date.month, date.day
                    ),
                    to=datetime.datetime(  # noqa: DTZ001
                        next_day.year, next_day.month, next_day.day
                    ),
                    type=type,
                    sensor_id="-1",
                )
                for type in DataValueType
            ]
            avg_data = [data[0] for data in avg_data_nested if len(data) > 0]
        except httpx.HTTPError:
            return f"Invalid city name `{city}`."

    if len(avg_data) == 0:
        return "No data available for this city."
    avg_data_str = ", ".join(
        f"{data_value.type}: {data_value.value}" for data_value in avg_data
    )

    return f"Air quality for `{city}` on `{date}`: {avg_data_str}"


@tool(
    "get_average_location_air_quality_on_date",
    args_schema=GetAverageLocationAirQualityOnDate,
)
def get_average_location_air_quality_on_date(  # noqa: PLR0914
    city: str, location: str, date: datetime.date
) -> str:
    """Get the day average air quality stats of a specific input location on a date."""

    if not location.casefold().endswith(city.casefold()):
        location = f"{location}, {city}"

    if date == datetime.datetime.now(tz=datetime.UTC).date():
        return (
            "Using this tool for today's date is not supported!"
            " You can only get daily average for a finished day."
        )
    next_day = date + datetime.timedelta(days=1)
    geocoder = OpenCageGeocode(OPENCAGE_API)
    results = geocoder.geocode(location)
    result = results[0]
    formatted_location = result["formatted"]
    target_location_lat_lng = result["geometry"]["lat"], result["geometry"]["lng"]
    pulse_eco_client = PulseEcoClient(city)

    sensors = pulse_eco_client.sensors()
    distances = [
        geopy.distance.geodesic(
            tuple(map(float, sensor.position.split(","))), target_location_lat_lng
        )
        for sensor in sensors
    ]
    sensors_and_distances_sorted = sorted(
        zip(distances, range(len(sensors)), sensors, strict=True)
    )
    for distance, _, sensor in sensors_and_distances_sorted:
        if distance.kilometers > 2:
            break
        from_ = datetime.datetime(  # noqa: DTZ001
            date.year, date.month, date.day
        )
        to = datetime.datetime(  # noqa: DTZ001
            next_day.year, next_day.month, next_day.day
        )
        raw_data = pulse_eco_client.data_raw(
            from_=from_, to=to, sensor_id=sensor.sensor_id
        )
        if len(raw_data) == 0:
            continue
        avg_data_nested = [
            pulse_eco_client.avg_data(
                period=AveragePeriod.DAY,
                from_=from_,
                to=to,
                type=type,
                sensor_id=sensor.sensor_id,
            )
            for type in DataValueType
        ]
        avg_data = [data[0] for data in avg_data_nested if len(data) > 0]
        if len(avg_data) == 0:
            continue
        avg_data_str = ", ".join(
            f"{data_value.type}: {data_value.value}" for data_value in avg_data
        )
        return (
            f"Input location resolved to `{formatted_location}`."
            f" Closest sensor to that location is `{sensor.description}`"
            f" with distance `{distance.kilometers:.3f} km`."
            f" The air quality for that sensor on `{date}`: {avg_data_str}"
        )
    return (
        f"Input location resolved to `{formatted_location}`."
        " No sensor was found in a 2 km radius of that location."
    )
