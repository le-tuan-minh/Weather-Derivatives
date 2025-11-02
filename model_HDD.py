import os
import pandas as pd
from datetime import datetime
import functions as f
import functions1 as f1  # giả định các module của bạn đã được import đúng

# === Cấu hình ===
STATION_FILE = "Station List Meteostat.xlsx"
DATA_FOLDER = "Weather Database Meteostat"
OUTPUT_FILE = "Weather_Index_HDD.xlsx"

# Các period cần tính
periods = [
    ("2024-11-01", "2024-11-30"),
    ("2024-12-01", "2024-12-31"),
    ("2025-01-01", "2025-01-31"),
    ("2025-02-01", "2025-02-28"),
    ("2025-03-01", "2025-03-31"),
    ("2024-11-01", "2025-03-31")
]

# === Hàm tính HDD cho 1 station ===
def compute_HDD_for_station(station_id, country):
    """
    Chạy toàn bộ pipeline mô phỏng và tính HDD forecast mean cho từng period.
    """
    try:
        file_path = os.path.join(DATA_FOLDER, f"{station_id}.csv")
        if not os.path.exists(file_path):
            print(f"[WARN] Không tìm thấy file cho station {station_id}, bỏ qua.")
            return None
        
        print(f"[INFO] Đang xử lý station {station_id} ({country})...")

        # Base temperature theo quốc gia
        base_temp = 18.33 if country == "USA" else 18

        # Bước 1: Tiền xử lý dữ liệu gốc
        df = f.preprocess(file_path, start="2020-10-01", end="2024-09-30")

        # Bước 2: Detrend & Deseasonalize
        df2, lambda_params = f1.detrend_deseasonalize(df, T_col='T', show_output=False)

        # Bước 3: Fit CAR model
        df3, ar_params = f1.fitCAR(df2, X_col='X', p=3, show_output=False)

        # Bước 4: Fit seasonal variance
        df4, sig_params = f1.seas_vol(df3, K=4, show_output=False)

        # Bước 5: Fit GH distribution
        gh = f1.fit_gh(df4, col='eps', show_output=False)

        # Bước 6: Simulate forecast (1 năm)
        sims = f1.simulate_forecast_T_using_GH(
            df4,
            ar_params=ar_params,
            sig_params=sig_params,
            lambda_params=lambda_params,
            gh_fit=gh,
            horizon=182,
            n_sims=10000,
            p=3,
            seed=123,
            plot=False
        )

        # Bước 7: Tính HDD cho từng period
        results = []
        for start, end in periods:
            hdd_val = f.calc_degree_index(
                sims['paths_df'],
                index_type='HDD',
                base_temp=base_temp,
                start=start,
                end=end
            ).mean()

            results.append({
                "Station ID": station_id,
                "Index": "HDD",
                "Start": start,
                "End": end,
                "Forecast": hdd_val
            })

        return results

    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý {station_id}: {e}")
        return None


# === Main script ===
def main():
    # Đọc danh sách trạm
    stations_df = pd.read_excel(STATION_FILE, dtype={"Station ID": str})

    all_results = []
    for _, row in stations_df.iterrows():
        station_id = row["Station ID"]
        country = row.get("Country", "UNKNOWN")
        res = compute_HDD_for_station(station_id, country)
        if res:
            all_results.extend(res)

    # Tổng hợp kết quả
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ Hoàn tất! Kết quả được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
