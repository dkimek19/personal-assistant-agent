"""Tests for the Open-Meteo weather tool (assistant.tools.weather).

Covers:
- _describe_weather_code: known codes mapped to descriptions, unknown -> "Unknown"
- _geocode_location: success, no-match (LocationNotFoundError), HTTP/connection
  errors, and request parameter forwarding
- _fetch_current_weather: success, missing current_weather block, HTTP errors,
  and request parameter forwarding
- get_current_weather:
  - returns a WeatherReport with data-fidelity passthrough of factual fields
  - empty/whitespace location raises ValueError
  - location is stripped before geocoding
  - LocationNotFoundError is raised immediately (non-retryable, no sleep)
  - transient errors are retried with exponential backoff (sleep 1.0, 2.0)
  - RuntimeError raised after max_retries exhausted, message mentions the
    location and attempt count
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from assistant.tools.weather import (
    LocationNotFoundError,
    WeatherReport,
    _describe_weather_code,
    _fetch_current_weather,
    _geocode_location,
    get_current_weather,
)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

GEOCODING_RESPONSE_SEOUL = {
    "results": [
        {
            "id": 1835848,
            "name": "Seoul",
            "latitude": 37.566,
            "longitude": 126.9784,
            "country": "South Korea",
            "admin1": "Seoul",
        }
    ]
}

GEOCODING_RESPONSE_EMPTY = {"results": []}

FORECAST_RESPONSE_SEOUL = {
    "latitude": 37.566,
    "longitude": 126.9784,
    "current_weather": {
        "temperature": 22.5,
        "windspeed": 8.3,
        "winddirection": 270,
        "weathercode": 1,
        "is_day": 1,
        "time": "2026-06-10T12:00",
    },
}


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    raise_for_status_exc: Exception | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body if json_body is not None else {}
    if raise_for_status_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Tests for _describe_weather_code
# ---------------------------------------------------------------------------


class TestDescribeWeatherCode:
    def test_clear_sky(self):
        assert _describe_weather_code(0) == "Clear sky"

    def test_slight_rain(self):
        assert _describe_weather_code(61) == "Slight rain"

    def test_thunderstorm(self):
        assert _describe_weather_code(95) == "Thunderstorm"

    def test_unknown_code_returns_unknown(self):
        assert _describe_weather_code(9999) == "Unknown"


# ---------------------------------------------------------------------------
# Tests for _geocode_location
# ---------------------------------------------------------------------------


class TestGeocodeLocation:
    @patch("assistant.tools.weather.httpx.get")
    def test_returns_resolved_name_and_coordinates(self, mock_get):
        mock_get.return_value = _make_response(json_body=GEOCODING_RESPONSE_SEOUL)

        name, lat, lon = _geocode_location("Seoul", base_url="https://geo.example", timeout=5.0)

        assert name == "Seoul"
        assert lat == 37.566
        assert lon == 126.9784

    @patch("assistant.tools.weather.httpx.get")
    def test_request_params_include_name_count_format(self, mock_get):
        mock_get.return_value = _make_response(json_body=GEOCODING_RESPONSE_SEOUL)

        _geocode_location("Seoul", base_url="https://geo.example", timeout=5.0)

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"]["name"] == "Seoul"
        assert call_kwargs["params"]["count"] == 1
        assert call_kwargs["params"]["format"] == "json"

    @patch("assistant.tools.weather.httpx.get")
    def test_empty_results_raises_location_not_found(self, mock_get):
        mock_get.return_value = _make_response(json_body=GEOCODING_RESPONSE_EMPTY)

        with pytest.raises(LocationNotFoundError):
            _geocode_location("Atlantis", base_url="https://geo.example", timeout=5.0)

    @patch("assistant.tools.weather.httpx.get")
    def test_missing_results_key_raises_location_not_found(self, mock_get):
        mock_get.return_value = _make_response(json_body={})

        with pytest.raises(LocationNotFoundError):
            _geocode_location("Atlantis", base_url="https://geo.example", timeout=5.0)

    @patch("assistant.tools.weather.httpx.get")
    def test_location_not_found_error_mentions_location(self, mock_get):
        mock_get.return_value = _make_response(json_body=GEOCODING_RESPONSE_EMPTY)

        with pytest.raises(LocationNotFoundError) as exc_info:
            _geocode_location("Atlantis", base_url="https://geo.example", timeout=5.0)

        assert "Atlantis" in str(exc_info.value)

    @patch("assistant.tools.weather.httpx.get")
    def test_http_error_propagates(self, mock_get):
        error_response = _make_response(
            status_code=500,
            raise_for_status_exc=httpx.HTTPStatusError(
                "500 Server Error", request=MagicMock(), response=MagicMock()
            ),
        )
        mock_get.return_value = error_response

        with pytest.raises(httpx.HTTPStatusError):
            _geocode_location("Seoul", base_url="https://geo.example", timeout=5.0)

    @patch("assistant.tools.weather.httpx.get")
    def test_connection_error_propagates(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            _geocode_location("Seoul", base_url="https://geo.example", timeout=5.0)


# ---------------------------------------------------------------------------
# Tests for _fetch_current_weather
# ---------------------------------------------------------------------------


class TestFetchCurrentWeather:
    @patch("assistant.tools.weather.httpx.get")
    def test_returns_current_weather_block(self, mock_get):
        mock_get.return_value = _make_response(json_body=FORECAST_RESPONSE_SEOUL)

        current = _fetch_current_weather(37.566, 126.9784, base_url="https://forecast.example", timeout=5.0)

        assert current == FORECAST_RESPONSE_SEOUL["current_weather"]

    @patch("assistant.tools.weather.httpx.get")
    def test_request_params_include_lat_lon_and_current_weather_flag(self, mock_get):
        mock_get.return_value = _make_response(json_body=FORECAST_RESPONSE_SEOUL)

        _fetch_current_weather(37.566, 126.9784, base_url="https://forecast.example", timeout=5.0)

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"]["latitude"] == 37.566
        assert call_kwargs["params"]["longitude"] == 126.9784
        assert call_kwargs["params"]["current_weather"] == "true"

    @patch("assistant.tools.weather.httpx.get")
    def test_missing_current_weather_key_returns_empty_dict(self, mock_get):
        mock_get.return_value = _make_response(json_body={"latitude": 1.0, "longitude": 2.0})

        current = _fetch_current_weather(1.0, 2.0, base_url="https://forecast.example", timeout=5.0)

        assert current == {}

    @patch("assistant.tools.weather.httpx.get")
    def test_http_error_propagates(self, mock_get):
        error_response = _make_response(
            status_code=503,
            raise_for_status_exc=httpx.HTTPStatusError(
                "503 Service Unavailable", request=MagicMock(), response=MagicMock()
            ),
        )
        mock_get.return_value = error_response

        with pytest.raises(httpx.HTTPStatusError):
            _fetch_current_weather(37.566, 126.9784, base_url="https://forecast.example", timeout=5.0)

    @patch("assistant.tools.weather.httpx.get")
    def test_timeout_error_propagates(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(httpx.TimeoutException):
            _fetch_current_weather(37.566, 126.9784, base_url="https://forecast.example", timeout=5.0)


# ---------------------------------------------------------------------------
# Tests for get_current_weather (success path)
# ---------------------------------------------------------------------------


class TestGetCurrentWeatherSuccess:
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_returns_weather_report_instance(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul")

        assert isinstance(report, WeatherReport)

    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_factual_fields_pass_through_unmodified(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul")

        assert report.location_name == "Seoul"
        assert report.latitude == 37.566
        assert report.longitude == 126.9784
        assert report.temperature_c == 22.5
        assert report.wind_speed_kmh == 8.3
        assert report.observation_time == "2026-06-10T12:00"

    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_weather_description_derived_from_code(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul")

        assert report.weather_code == 1
        assert report.weather_description == "Mainly clear"

    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_to_dict_contains_all_ontology_keys(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul")
        d = report.to_dict()

        assert set(d.keys()) == {
            "location_name",
            "latitude",
            "longitude",
            "temperature_c",
            "wind_speed_kmh",
            "weather_code",
            "weather_description",
            "observation_time",
        }

    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_location_stripped_before_geocoding(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        get_current_weather("  Seoul  ")

        called_location = mock_geocode.call_args.args[0]
        assert called_location == "Seoul"

    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_coordinates_forwarded_from_geocode_to_fetch(self, mock_geocode, mock_fetch):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        get_current_weather("Seoul")

        call_args = mock_fetch.call_args.args
        assert call_args[0] == 37.566
        assert call_args[1] == 126.9784

    def test_empty_location_raises_value_error(self):
        with pytest.raises(ValueError):
            get_current_weather("")

    def test_whitespace_only_location_raises_value_error(self):
        with pytest.raises(ValueError):
            get_current_weather("   ")


# ---------------------------------------------------------------------------
# Tests for get_current_weather (error / retry handling)
# ---------------------------------------------------------------------------


class TestGetCurrentWeatherErrors:
    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_location_not_found_raised_immediately_without_retry(
        self, mock_geocode, mock_fetch, mock_sleep
    ):
        mock_geocode.side_effect = LocationNotFoundError("No location found matching 'Atlantis'")

        with pytest.raises(LocationNotFoundError):
            get_current_weather("Atlantis")

        assert mock_geocode.call_count == 1
        mock_fetch.assert_not_called()
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_retries_on_transient_geocode_error_and_succeeds(
        self, mock_geocode, mock_fetch, mock_sleep
    ):
        mock_geocode.side_effect = [
            httpx.ConnectError("Connection refused"),
            ("Seoul", 37.566, 126.9784),
        ]
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul", max_retries=2, base_backoff=1.0)

        assert report.location_name == "Seoul"
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_retries_on_transient_forecast_error_and_succeeds(
        self, mock_geocode, mock_fetch, mock_sleep
    ):
        mock_geocode.return_value = ("Seoul", 37.566, 126.9784)
        mock_fetch.side_effect = [
            httpx.TimeoutException("Request timed out"),
            FORECAST_RESPONSE_SEOUL["current_weather"],
        ]

        report = get_current_weather("Seoul", max_retries=2, base_backoff=1.0)

        assert report.temperature_c == 22.5
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_second_retry_uses_doubled_backoff(self, mock_geocode, mock_fetch, mock_sleep):
        mock_geocode.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            ("Seoul", 37.566, 126.9784),
        ]
        mock_fetch.return_value = FORECAST_RESPONSE_SEOUL["current_weather"]

        report = get_current_weather("Seoul", max_retries=2, base_backoff=1.0)

        assert report.location_name == "Seoul"
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0
        assert calls[1][0][0] == 2.0

    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_raises_runtime_error_after_max_retries(self, mock_geocode, mock_fetch, mock_sleep):
        mock_geocode.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            get_current_weather("Seoul", max_retries=2, base_backoff=0.01)

        error_msg = str(exc_info.value)
        assert "Unable to retrieve weather" in error_msg
        assert "Seoul" in error_msg
        assert "3 attempts" in error_msg
        assert "Connection refused" in error_msg

    @patch("time.sleep", return_value=None)
    @patch("assistant.tools.weather._fetch_current_weather")
    @patch("assistant.tools.weather._geocode_location")
    def test_geocode_called_exactly_max_retries_plus_one_times(
        self, mock_geocode, mock_fetch, mock_sleep
    ):
        mock_geocode.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(RuntimeError):
            get_current_weather("Seoul", max_retries=2, base_backoff=0.01)

        assert mock_geocode.call_count == 3
        mock_fetch.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
