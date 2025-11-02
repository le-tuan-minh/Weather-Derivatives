import os
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Daily

STATION_FILE = "Station List Meteostat.xlsx"
OUTPUT_FOLDER = "Weather Database Meteostat"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Đọc station ID dạng text
stations_df = pd.read_excel(STATION_FILE, dtype={"Station ID": str})
stations_df["Station ID"] = stations_df["Station ID"].str.strip()

start_date = datetime(2004, 1, 1)
yesterday = datetime.today() - timedelta(days=1)  # datetime object


def fetch_and_save_station(station_id: str):
    station_id = str(station_id).strip()
    file_path = os.path.join(OUTPUT_FOLDER, f"{station_id}.csv")

    # ==== Cập nhật dữ liệu ====
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, parse_dates=["time"])
        existing.set_index("time", inplace=True)

        update_start = yesterday - timedelta(days=30)
        if update_start < start_date:
            update_start = start_date
        update_end = yesterday
        old_data = existing.loc[existing.index < update_start]
    else:
        update_start = start_date
        update_end = yesterday
        old_data = pd.DataFrame()

    # Tải dữ liệu mới (model=True)
    try:
        data = Daily(station_id, update_start, update_end, model=True).fetch()
    except Exception as e:
        print(f"Error fetching data for {station_id}:", e)
        return

    # Giữ lại cột time, tmin, tmax
    data = data.reset_index()[["time", "tmin", "tmax"]]
    data.set_index("time", inplace=True)

    if not old_data.empty:
        df = pd.concat([old_data, data])
    else:
        df = data

    # Loại bỏ dòng trùng theo thời gian, giữ bản ghi cuối
    df = df[~df.index.duplicated(keep="last")]

    # Lưu file CSV với định dạng ngày chuẩn
    df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
    print(f"Saved {len(df)} rows to {file_path}")


def main():
    for _, row in stations_df.iterrows():
        station_id = row["Station ID"]
        if pd.notna(station_id):
            fetch_and_save_station(station_id)

    # Không còn ghi Missing & Missing Rate vào Excel nữa
    print("All stations updated.")


if __name__ == "__main__":
    main()



