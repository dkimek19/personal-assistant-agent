"""Open-Meteo weather tool for the Personal Assistant Agent.

Provides one public function:
- get_current_weather: Resolves a location name to coordinates via the
  Open-Meteo Geocoding API, then fetches current weather conditions via the
  Open-Meteo Forecast API. Both APIs are free and require no API key.

Tool-returned factual data (location name, coordinates, temperature, wind
speed, weather code/description, observation time) is never altered by the
LLM - formatting only, no content modification.

Constraint: exponential backoff with 2 retries on transient API failures,
then an explicit RuntimeError for the user. A location name that the
geocoding API cannot resolve is treated as a non-retryable input error
(LocationNotFoundError) since retrying would not change the result.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEOCODING_URL: str = os.environ.get(
    "OPEN_METEO_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
FORECAST_URL: str = os.environ.get(
    "OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
)

_REQUEST_TIMEOUT_SECONDS: float = 10.0

# WMO weather interpretation codes used by Open-Meteo, mapped to
# human-readable descriptions.
_WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LocationNotFoundError(ValueError):
    """Raised when the geocoding API returns no match for a location name.

    Non-retryable: a bad location name will not become valid by retrying.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WeatherReport:
    """Structured current-weather conditions for a resolved location.

    All factual fields are stored exactly as returned by the Open-Meteo API
    (or derived via a fixed lookup table for ``weather_description``) - no
    LLM modification allowed.
    """

    location_name: str
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kmh: float
    weather_code: int
    weather_description: str
    observation_time: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (safe to pass to LLM context)."""
        return {
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature_c": self.temperature_c,
            "wind_speed_kmh": self.wind_speed_kmh,
            "weather_code": self.weather_code,
            "weather_description": self.weather_description,
            "observation_time": self.observation_time,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe_weather_code(code: int) -> str:
    """Translate a WMO weather interpretation code to a human-readable string."""
    return _WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown")


def _geocode_location(
    location: str, *, base_url: str, timeout: float
) -> tuple[str, float, float]:
    """Resolve a location name to ``(resolved_name, latitude, longitude)``.

    Raises:
        LocationNotFoundError: If the geocoding API returns no matches.
        httpx.HTTPStatusError: On a non-2xx HTTP response (retryable).
        httpx.RequestError: On a connection/timeout error (retryable).
    """
    response = httpx.get(
        base_url,
        params={"name": location, "count": 1, "format": "json"},
        timeout=timeout,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    results: list[dict[str, Any]] = data.get("results") or []
    if not results:
        raise LocationNotFoundError(f"No location found matching '{location}'")

    top = results[0]
    return str(top.get("name", location)), float(top["latitude"]), float(top["longitude"])


def _fetch_current_weather(
    latitude: float, longitude: float, *, base_url: str, timeout: float
) -> dict[str, Any]:
    """Fetch the ``current_weather`` block from the Open-Meteo forecast API.

    Raises:
        httpx.HTTPStatusError: On a non-2xx HTTP response (retryable).
        httpx.RequestError: On a connection/timeout error (retryable).
    """
    response = httpx.get(
        base_url,
        params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
        timeout=timeout,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    return data.get("current_weather") or {}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def get_current_weather(
    location: str,
    *,
    geocoding_url: str = GEOCODING_URL,
    forecast_url: str = FORECAST_URL,
    max_retries: int = 2,
    base_backoff: float = 1.0,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
) -> WeatherReport:
    """Get current weather conditions for a named location.

    Parameters
    ----------
    location:
        A place name, e.g. ``"Seoul"`` or ``"Paris, France"``. Resolved to
        coordinates via the Open-Meteo Geocoding API.
    geocoding_url:
        Base URL of the Open-Meteo geocoding endpoint.
    forecast_url:
        Base URL of the Open-Meteo forecast endpoint.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    WeatherReport
        Current weather conditions for the resolved location.

    Raises
    ------
    ValueError
        If *location* is empty or whitespace-only.
    LocationNotFoundError
        If the geocoding API cannot resolve *location* to coordinates.
        Raised immediately, without retries.
    RuntimeError
        After *max_retries* transient failures, a ``RuntimeError`` is raised
        with an explicit human-readable message for the user.
    """
    if not location or not location.strip():
        raise ValueError("get_current_weather() requires a non-empty location string")

    location = location.strip()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Open-Meteo retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            resolved_name, latitude, longitude = _geocode_location(
                location, base_url=geocoding_url, timeout=timeout
            )
            current = _fetch_current_weather(
                latitude, longitude, base_url=forecast_url, timeout=timeout
            )

            weather_code = int(current.get("weathercode", -1))
            logger.info("Retrieved current weather for '%s'", resolved_name)
            return WeatherReport(
                location_name=resolved_name,
                latitude=latitude,
                longitude=longitude,
                temperature_c=float(current.get("temperature", 0.0)),
                wind_speed_kmh=float(current.get("windspeed", 0.0)),
                weather_code=weather_code,
                weather_description=_describe_weather_code(weather_code),
                observation_time=str(current.get("time", "")),
            )

        except LocationNotFoundError:
            raise

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error("Open-Meteo API error on attempt %d: %s", attempt + 1, exc)

    raise RuntimeError(
        f"Unable to retrieve weather for '{location}' after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your network connection and try again."
    )
