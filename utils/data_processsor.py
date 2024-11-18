from typing import List
import pandas as pd
from PIL import Image
from pi_heif import register_heif_opener
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


register_heif_opener()


def get_geolocation_from_postcode(postcode: str):
    if not postcode:
        return None, None
    try:
        geolocator = Nominatim(user_agent="spendingHabits")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geocode(postcode)
        return location.latitude, location.longitude
    except Exception as e:
        print(f"Error getting location from postcode {postcode}: {e}")
        return None, None


def get_image_host_make_and_model(image_path):
    if not image_path:
        return ""
    from PIL.ExifTags import TAGS

    im = Image.open(image_path)  # do whatever need with a Pillow image
    # Extract EXIF data
    exif_data = im.getexif()
    return f"{exif_data.get(271, '')} {exif_data.get(272, '')}"


def week_of_month(date):
    first_day = date.replace(day=1)
    # Calculate the week by checking how many full 7-day periods have passed
    adjusted_day = date.day + first_day.weekday()
    return (adjusted_day - 1) // 7 + 1


def get_dataframe_from_data(data: List[dict], image_paths: List[str]):
    df = pd.DataFrame(data)

    # add lat and long based on postal code
    df["latitude"], df["longitude"] = zip(
        *df["businessPostalCode"].apply(get_geolocation_from_postcode)
    )
    df["image_path"] = image_paths
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="mixed", dayfirst=True
    )
    df["paymentMethod"] = df["paymentMethod"].fillna("card")
    # based on date time add a column for the day of the  week  ie (Mon,Tues .etc) and another for week or weekend also add day of the month
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["weekend"] = df["datetime"].dt.weekday // 5
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_month"] = df["datetime"].apply(week_of_month)
    df["month"] = df["datetime"].dt.month_name()
    df["year"] = df["datetime"].dt.year
    # add a column that classifies the date time as morning afternoon or evening
    df["time_of_day"] = pd.cut(
        df["datetime"].dt.hour,
        bins=[0, 12, 18, 24],
        labels=["morning", "afternoon", "evening"],
        right=False,
    )
    df["device"] = df["image_path"].apply(get_image_host_make_and_model)

    return df


def add_one_row_to_dataframe(
    df: pd.DataFrame, data: dict, image_path: str
) -> pd.DataFrame:
    row = get_dataframe_from_data([data], [image_path])
    return pd.concat([df, row], ignore_index=True)
