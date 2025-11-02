import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

# ---------------------------
# preprocess (giữ nguyên signature)
# ---------------------------
def preprocess(data, start=None, end=None):
    # Load
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data phải là đường dẫn csv hoặc pandas.DataFrame")
    
    # Đổi tên cột
    rename_map = {'time': 'DATE', 'tmax': 'TMAX', 'tmin': 'TMIN'}
    df = df.rename(columns=rename_map)
    
    # Chuyển DATE thành datetime
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)

    # Lọc theo khoảng thời gian nếu có
    if start is not None:
        start = pd.to_datetime(start)
        df = df[df['DATE'] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        df = df[df['DATE'] <= end]
    df = df.reset_index(drop=True)

    # --- Thêm bước kiểm tra và điền ngày bị thiếu ---
    full_range = pd.date_range(df['DATE'].min(), df['DATE'].max(), freq='D')
    df_full = pd.DataFrame({'DATE': full_range})
    df = pd.merge(df_full, df, on='DATE', how='left')

    # --- Tính T ---
    df['T'] = (df['TMAX'].astype(float) + df['TMIN'].astype(float)) / 2.0

    # Hàm fill giá trị thiếu cho T (mean trong cửa sổ ±7, không dùng giá trị đang fill)
    def fill_missing_T(series):
        s = series.copy()
        nan_idx = np.where(s.isna())[0]
        for idx in nan_idx:
            start_idx = max(0, idx - 7)
            end_idx = min(len(s) - 1, idx + 7)
            window = s.iloc[start_idx:end_idx+1].drop(index=idx, errors='ignore')
            window = window[window.notna()]
            if len(window) > 0:
                s.iat[idx] = window.mean()
        return s

    df['T'] = fill_missing_T(df['T'])

    # --- Tách ngày/tháng/năm ---
    df['DAY'] = df['DATE'].dt.day
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year

    # --- Bỏ ngày 29/02 nếu có ---
    leap_mask = (df['MONTH'] == 2) & (df['DAY'] == 29)
    if leap_mask.any():
        df = df.loc[~leap_mask].reset_index(drop=True)
        
    # --- Chỉ giữ các cột cần thiết ---
    df = df[['DATE', 'TMAX', 'TMIN', 'T', 'DAY', 'MONTH', 'YEAR']]
    
    return df



# ---------------------------
# detrend_deseasonalize (giữ signature, sửa label thành LAMBDA(t))
# ---------------------------
def detrend_deseasonalize(df, T_col='T', LAMBDA_col='LAMBDA', X_col='X',
                          omega=2 * np.pi / 365, return_results=False):
    if T_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{T_col}' trong df")
    n = len(df)
    if n == 0:
        raise ValueError("DataFrame rỗng")

    # thời gian
    t = np.arange(n, dtype=float)

    # Vẽ line plot ban đầu
    plt.figure(figsize=(10, 4))
    plt.plot(t, df[T_col], label='Observed T', linestyle='-', marker='', alpha=0.8)
    plt.title('Chuỗi gốc trước khi fit')
    plt.xlabel('t')
    plt.ylabel(T_col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Ma trận thiết kế
    X = pd.DataFrame({
        'const': 1.0,
        't': t,
        'cos_wt': np.cos(omega * t),
        'sin_wt': np.sin(omega * t)
    }, index=df.index)

    # Fit OLS
    y = df[T_col].astype(float).values
    model = sm.OLS(y, X)
    results = model.fit()

    # Lưu fitted & residual
    df[LAMBDA_col] = results.predict(X)
    df[X_col] = df[T_col] - df[LAMBDA_col]

    # Vẽ scatter thực tế + line fitted với label LAMBDA(t)
    plt.figure(figsize=(10, 4))
    plt.scatter(t, df[T_col], label='Observed T', s=10, alpha=0.6)
    plt.plot(t, df[LAMBDA_col], label='Λ(t)', color='red', linewidth=2)
    plt.title('Dữ liệu thực tế (scatter) và fitted (line)')
    plt.xlabel('t')
    plt.ylabel(T_col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # In kết quả hồi quy
    print(results.summary())

    return results if return_results else None


# ---------------------------
# ADF detailed (giữ nguyên)
# ---------------------------
def adf_detailed(y, regression='c'):
    y = y.dropna()

    # 1. Sử dụng adfuller để chọn số trễ theo AIC và lấy critical values
    result = adfuller(y, regression=regression, autolag='AIC')
    best_lag = result[2]
    critical_values = result[4]
    adf_statistic = result[0]

    print(f"\n=== ADF regression: '{regression}' | optimal lag (by AIC) = {best_lag} ===")

    # 2. Dựng lại mô hình OLS để lấy hệ số và p-value
    dy = y.diff().dropna()
    y_lag1 = y.shift(1).dropna().loc[dy.index]

    if best_lag > 0:
        dy_lags = pd.concat([dy.shift(i) for i in range(1, best_lag + 1)], axis=1)
        dy_lags.columns = [f'dy_lag{i}' for i in range(1, best_lag + 1)]
        dy_lags = dy_lags.dropna()
        y_lag1 = y_lag1.loc[dy_lags.index]
        dy = dy.loc[y_lag1.index]
        X = pd.concat([y_lag1.rename('y_lag1'), dy_lags], axis=1)
    else:
        dy = dy.loc[y_lag1.index]
        X = pd.DataFrame({'y_lag1': y_lag1})

    if regression == 'c':
        X = sm.add_constant(X)
    elif regression == 'ct':
        X['trend'] = np.arange(1, len(X) + 1)
        X = sm.add_constant(X)
    elif regression != 'n':
        raise ValueError("regression must be 'c', 'ct', or 'n'")

    model = sm.OLS(dy, X).fit()

    # 3. In hệ số và p-value
    for var in model.params.index:
        coef = model.params[var]
        pval = model.pvalues[var]
        print(f"{var:<10} | Coef: {coef: .4f} | p-value: {pval: .4f}")

    # 4. In ADF statistic và critical values
    print(f"\nADF test statistic: {adf_statistic:.4f}")
    print("Critical values:")
    for level, value in critical_values.items():
        print(f"  {level}: {value:.4f}")


def adf_and_acf_pacf(df):
    adf_detailed(df['X'], regression='ct')  # Trend + drift
    adf_detailed(df['X'], regression='c')   # Drift only
    adf_detailed(df['X'], regression='n')   # No trend or drift
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(df['X'], ax=axes[0], lags=365, markersize=1)
    plot_pacf(df['X'], ax=axes[1], lags=10, method='ywm')

    axes[0].set_title("ACF of Residuals")
    axes[1].set_title("PACF of Residuals")
    plt.tight_layout()
    plt.show()


# ---------------------------
# fitCAR (giữ nguyên tên + defaults)
# ---------------------------
def fitCAR(df, 
           X_col='X', 
           p=3, 
           date_col='DATE', 
           pacf_lag=10, 
           acf_lag=365):
    """
    Fit mô hình AR(p) trên X_col, tạo residuals u và u_sq.
    In summary AR và plot ACF/PACF cho u và u_sq.
    """
    if X_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{X_col}' trong df")

    # Sắp xếp theo thời gian nếu có date_col
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df.sort_values(by=date_col, inplace=True)

    # Chuẩn bị y
    y = df[X_col].astype(float)
    y_nonan = y.dropna().copy()
    if len(y_nonan) <= p:
        raise ValueError(f"Không đủ quan sát để fit AR(p): {len(y_nonan)} <= p={p}")

    # Ép index về RangeIndex để tránh ValueWarning (giữ alignment)
    original_idx = y_nonan.index.to_numpy()
    y_vals = y_nonan.values

    ar_model = AutoReg(y_vals, lags=p, trend='n', old_names=False)
    ar_res = ar_model.fit()
    fitted = ar_res.fittedvalues
    resid = ar_res.resid  # length = len(y_vals) - p

    # Gán residuals trở lại df: residuals tương ứng với positions original_idx[p:]
    start_pos = p
    resid_positions = original_idx[start_pos:]
    df = df.copy()
    df['u'] = np.nan
    df.loc[resid_positions, 'u'] = resid
    df['u_sq'] = df['u'] ** 2

    # In summary gọn
    print("=== AR Model Summary ===")
    print(ar_res.summary())

    # Plot ACF/PACF cho u và u_sq
    def plot_corr(series, name):
        s = series.dropna()
        if len(s) == 0:
            print(f"No data to plot ACF/PACF for {name}")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sm.graphics.tsa.plot_acf(s, lags=acf_lag, ax=axes[0], markersize=1)
        sm.graphics.tsa.plot_pacf(s, lags=pacf_lag, ax=axes[1])
        axes[0].set_title(f'ACF of {name}')
        axes[1].set_title(f'PACF of {name}')
        plt.tight_layout()
        plt.show()

    plot_corr(df['u'], 'u')
    plot_corr(df['u_sq'], 'u_sq')

    return df


# ---------------------------
# seas_vol (giữ nguyên tên + defaults)
# ---------------------------
def seas_vol(df, 
             K=4, 
             date_col='DATE', 
             omega=2 * np.pi / 365,
             clip_sigma_sq_min=1e-12,
             plot_last_n=365,
             show_summaries=True):
    """
    Fit seasonal variance model dựa trên u_sq_mean theo DAY.
    Thêm cột sigma_sq và eps.
    """
    df = df.copy()
    # Đảm bảo có MONTH và DAY để groupby
    if 'MONTH' not in df.columns and date_col in df.columns:
        df['MONTH'] = df[date_col].dt.month
    if 'DAY' not in df.columns:
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['DAY'] = df[date_col].dt.day
        else:
            df['DAY'] = (np.arange(len(df)) % 31) + 1

    # u_sq_mean theo groupby (MONTH, DAY)
    if 'u_sq' not in df.columns:
        df['u_sq'] = np.nan
    df['u_sq_mean'] = df.groupby(['MONTH', 'DAY'])['u_sq'].transform('mean')

    # Design matrix cos/sin
    t_seq = np.arange(len(df), dtype=float)
    X_sig = pd.DataFrame({'const': 1}, index=df.index)
    for k in range(1, K+1):
        X_sig[f'cos_{k}'] = np.cos(omega * k * t_seq)
        X_sig[f'sin_{k}'] = np.sin(omega * k * t_seq)

    y_sig = df['u_sq_mean']
    mask = y_sig.notna() & X_sig.notna().all(axis=1)

    if mask.sum() == 0:
        df['sigma_sq'] = np.nan
        df['eps'] = np.nan
        if show_summaries:
            print("⚠ Không có dữ liệu hợp lệ để fit seasonal variance.")
        return df, None

    sig_model = sm.OLS(y_sig[mask].values, X_sig[mask].values)
    sig_res = sig_model.fit()

    if show_summaries:
        print("=== Seasonal Variance Model Summary ===")
        print(sig_res.summary())

    sigma_sq_fitted = pd.Series(np.nan, index=df.index)
    sigma_sq_fitted.loc[mask] = sig_res.predict(X_sig[mask].values)
    sigma_sq_fitted = sigma_sq_fitted.clip(lower=clip_sigma_sq_min)
    df['sigma_sq'] = sigma_sq_fitted

    sigma = np.sqrt(df['sigma_sq'])
    df['eps'] = (df['u'] / sigma).replace([np.inf, -np.inf], np.nan)

    # Plot sigma_sq vs u_sq_mean (last n)
    last_n = min(plot_last_n, len(df))
    df_plot = df.iloc[-last_n:]
    if date_col in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
        x = df_plot[date_col]
        xlabel = 'DATE'
    else:
        x = df_plot.index
        xlabel = 'index'

    plt.figure(figsize=(10, 4))
    plt.plot(x, df_plot['u_sq_mean'], label='u_sq_mean (obs)', linewidth=1)
    plt.plot(x, df_plot['sigma_sq'], label='sigma_sq (fitted)', linewidth=1)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Value')
    plt.title(f'Seasonal variance fit (last {last_n} obs)')
    plt.tight_layout()
    plt.show()

    return df


# ---------------------------
# check_residuals (giữ nguyên)
# ---------------------------
def check_residuals(df, eps_col='eps', lags=10, title_city=None):
    if eps_col not in df.columns:
        raise ValueError(f"Cột '{eps_col}' không tồn tại trong DataFrame")
    
    eps = df[eps_col].dropna()
    
    print("=== 📊 Kiểm tra phân phối chuẩn ===")
    jb_stat, jb_p = jarque_bera(eps)
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
    
    print("\n=== 🔍 Kiểm tra tự tương quan (Ljung–Box) ===")
    lb_res = acorr_ljungbox(eps, lags=[lags], return_df=True)
    print(f"Ljung–Box test for {eps_col} (lag={lags}): statistic={lb_res['lb_stat'].iloc[0]:.4f}, "
          f"p-value={lb_res['lb_pvalue'].iloc[0]:.4f}")
    
    # ACF & PACF cho eps
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    plot_acf(eps, ax=ax[0], lags=365, markersize=1)
    plot_pacf(eps, ax=ax[1], lags=lags, method='ywm')
    ax[0].set_title(f'ACF ({title_city})')
    ax[1].set_title(f'PACF ({title_city})')
    plt.show()
    
    # --- Cho eps^2 ---
    eps2 = eps**2
    
    print("\n=== 🔍 Kiểm tra tự tương quan (Ljung–Box) cho bình phương ===")
    lb_res2 = acorr_ljungbox(eps2, lags=[lags], return_df=True)
    print(f"Ljung–Box test for {eps_col}² (lag={lags}): statistic={lb_res2['lb_stat'].iloc[0]:.4f}, "
          f"p-value={lb_res2['lb_pvalue'].iloc[0]:.4f}")
    
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    plot_acf(eps2, ax=ax[0], lags=365, markersize=1)
    plot_pacf(eps2, ax=ax[1], lags=lags, method='ywm')
    ax[0].set_title(f'ACF ({title_city})')
    ax[1].set_title(f'PACF ({title_city})')
    plt.show()


# ---------------------------
# fit_gh (giữ nguyên tên + behavior)
# ---------------------------
def fit_gh(df, col='eps'):
    try:
        from rpy2.robjects import r, FloatVector
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as e:
        raise ImportError("rpy2 không khả dụng. Cài đặt rpy2 + package R 'ghyp' nếu cần fit GH.") from e

    # Import R package once
    ghyp = importr("ghyp")

    eps = df[col].dropna()
    if eps.empty:
        raise ValueError("Không có dữ liệu để fit GH.")

    with localconverter(pandas2ri.converter):
        eps_r = pandas2ri.py2rpy(eps)
    fit = r['fit.ghypuv'](eps_r, silent=True)

    params_r = r['coef'](fit)
    params_dict = {name: np.array(params_r.rx2(name))[0] for name in params_r.names}
    print("✅ Estimated GH parameters:")
    for k, v in params_dict.items():
        print(f"  {k}: {v:.6f}")

    x_grid = np.linspace(eps.min(), eps.max(), 500)
    x_r = FloatVector(x_grid)
    pdf_vals = r['dghyp'](x_r, fit)
    with localconverter(pandas2ri.converter):
        pdf_py = np.array(pandas2ri.rpy2py(pdf_vals))

    plt.figure(figsize=(8, 5))
    plt.hist(eps, bins=50, density=True, alpha=0.5, color="skyblue", label="Data histogram")
    plt.plot(x_grid, pdf_py, 'r-', lw=2, label="Fitted GH pdf")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.title("GH fit vs data")
    plt.legend()
    plt.show()

    return params_dict

import pandas as pd
import numpy as np

def calc_degree_index(df, start, end, base_temp=18.33, index_type="HDD"):
    """
    Tính HDD, CDD hoặc CAT tích lũy cho từng path trong khoảng thời gian chỉ định.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gồm cột 'DATE' và các cột 'path_*' (nhiệt độ mô phỏng).
    start, end : str hoặc datetime
        Ngày bắt đầu và kết thúc (inclusive).
    base_temp : float, mặc định 18.33
        Nhiệt độ cơ sở (dùng cho HDD và CDD).
    index_type : str, một trong {"HDD", "CDD", "CAT"}
        - HDD: sum(max(0, base_temp - T))
        - CDD: sum(max(0, T - base_temp))
        - CAT: sum(T)
    
    Returns
    -------
    pd.Series
        Giá trị chỉ số tích lũy cho từng path.
    """
    if "DATE" not in df.columns:
        raise ValueError("DataFrame phải có cột 'DATE'")

    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    df_period = df[(df["DATE"] >= start_ts) & (df["DATE"] <= end_ts)]
    path_cols = [c for c in df.columns if c.startswith("path_")]
    if not path_cols:
        raise ValueError("Không tìm thấy cột path_i nào trong DataFrame.")

    idx = index_type.upper()
    if idx == "HDD":
        result = (base_temp - df_period[path_cols]).clip(lower=0).sum()
    elif idx == "CDD":
        result = (df_period[path_cols] - base_temp).clip(lower=0).sum()
    elif idx == "CAT":
        result = df_period[path_cols].sum()
    else:
        raise ValueError("index_type phải là 'HDD', 'CDD' hoặc 'CAT'.")

    result.name = f"{idx}_{start_ts.date()}_{end_ts.date()}"
    return result
