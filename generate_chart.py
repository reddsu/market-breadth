import io
import os
import time
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# None 代表完整下載羅素 2000 全部成分股
RUSSELL_MAX_STOCKS = None

FALLBACK_SOX = [
    "AMD", "ADI", "AMAT", "ARM", "ASML", "AVGO", "COHR", "ENTG", "GFS", "INTC",
    "KLAC", "LRCX", "LSCC", "MCHP", "MPWR", "MRVL", "MU", "NVDA", "NXPI", "ON",
    "QCOM", "QRVO", "RMBS", "SWKS", "TSM", "TXN", "TER", "WDC", "WOLF", "AMKR",
]


def parse_ishares_csv_text(text, max_count=None):
    """解析 iShares CSV 內容，支援 BOM、引號表頭並防範 HTML 阻擋頁"""
    head_sample = text[:1000].lower()
    if "<html" in head_sample or "<!doctype" in head_sample:
        raise ValueError("回傳內容為 HTML 網頁而非 CSV（iShares 可能阻擋了雲端 IP）")

    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        cleaned_line = line.lstrip("\ufeff").strip()
        if cleaned_line.startswith("Ticker,") or cleaned_line.startswith('"Ticker",'):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("無法在 CSV 中找到 Ticker 表頭列")

    df = pd.read_csv(io.StringIO("\n".join(lines[start_idx:])), on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip().str.lower() == "equity"]

    raw_tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
    ticker_map = {
        "BRKB": "BRK-B", "BFB": "BF-B", "CRDA": "CRD-A", "CRDB": "CRD-B",
        "BIOB": "BIO-B", "HEIA": "HEI-A", "LENB": "LEN-B", "MOGA": "MOG-A",
    }

    cleaned = []
    for t in raw_tickers:
        if t in ("-", "USD", "IWM", "IVV", "SOXX", "XTSLA") or len(t) > 6:
            continue
        t_clean = ticker_map.get(t, t.replace(".", "-").replace(" ", "-"))
        if all(c.isalnum() or c == "-" for c in t_clean):
            cleaned.append(t_clean)

    cleaned = list(dict.fromkeys(cleaned))
    return cleaned[:max_count] if max_count else cleaned


def fetch_ishares_etf_tickers(csv_url, max_count=None):
    """從 iShares 線上抓取 CSV（快速失敗機制，避免浪費等待時間）"""
    res = requests.get(csv_url, headers=HEADERS, timeout=15)
    res.raise_for_status()
    text = res.content.decode("utf-8-sig", errors="replace")
    return parse_ishares_csv_text(text, max_count=max_count)


def get_all_constituents():
    print("1. 正在獲取三大指數成分股清單...")

    # (1) S&P 500 (iShares IVV -> GitHub 開源 CSV 備援)
    try:
        ivv_url = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
        sp500 = fetch_ishares_etf_tickers(ivv_url, max_count=503)
        if len(sp500) < 400:
            raise ValueError("IVV 成分股數量異常")
        print(f" - S&P 500: 成功從 iShares 線上取得 ({len(sp500)} 檔)")
    except Exception:
        sp500_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        sp500 = pd.read_csv(sp500_url)["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f" - S&P 500: 已透過開源 CSV 備援取得 ({len(sp500)} 檔)")

    # (2) Russell 2000 (iShares IWM -> 專案內本地 IWM_holdings.csv 備援)
    try:
        iwm_url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        r2000 = fetch_ishares_etf_tickers(iwm_url, max_count=RUSSELL_MAX_STOCKS)
        if len(r2000) < 500:
            raise ValueError("IWM 成分股數量異常")
        print(f" - Russell 2000: 成功從 iShares 線上取得 ({len(r2000)} 檔)")
    except Exception as err:
        local_iwm = next(
            (f for f in os.listdir(".") if f.lower().endswith(".csv") and "iwm" in f.lower()),
            None,
        )
        if not local_iwm:
            raise RuntimeError(
                f"無法從 iShares 線上抓取 Russell 2000 ({err})，且專案內找不到本地 IWM_holdings.csv 檔案。"
                "請用瀏覽器下載 IWM_holdings.csv 並上傳至 GitHub 專案根目錄。"
            )
        with open(local_iwm, "r", encoding="utf-8-sig", errors="replace") as f:
            r2000 = parse_ishares_csv_text(f.read(), max_count=RUSSELL_MAX_STOCKS)
        print(f" - Russell 2000: iShares 線上受限，已改用專案內 {local_iwm} 取得 ({len(r2000)} 檔)")

    # (3) SOX 費半 (iShares SOXX -> 內建清單備援)
    try:
        soxx_url = "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund"
        sox = fetch_ishares_etf_tickers(soxx_url, max_count=30)
        if len(sox) < 25:
            raise ValueError("SOXX 成分股數量異常")
        print(f" - SOX 費半: 成功從 iShares 線上取得 ({len(sox)} 檔)")
    except Exception:
        sox = FALLBACK_SOX
        print(f" - SOX 費半: 已切換內建清單備援 ({len(sox)} 檔)")

    return sp500, r2000, sox


def fetch_index_close(symbol):
    """抓取大盤指數本身收盤價，用於雙 Y 軸背離對照"""
    raw = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna()


def calc_breadth(tickers, name, batch_size=350):
    print(f" - 正在下載 {name} 報價並計算 20/50/200MA 寬度 (共 {len(tickers)} 檔)...")
    all_close = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(
            batch, period="2y", auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        elif "Close" in raw.columns:
            close_df = raw["Close"].to_frame(name=batch[0])
        else:
            close_df = raw

        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(name=batch[0])

        close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
        all_close.append(close_df)
        time.sleep(1)

    if not all_close:
        raise RuntimeError(f"{name} 下載資料為空")

    data = pd.concat(all_close, axis=1)
    data = data.loc[:, ~data.columns.duplicated()].dropna(axis=1, how="all")
    data = data[data.notna().sum(axis=1) > (len(tickers) * 0.5)]

    ma20 = data.rolling(window=20, min_periods=20).mean()
    ma50 = data.rolling(window=50, min_periods=50).mean()
    ma200 = data.rolling(window=200, min_periods=200).mean()

    pct_20 = ((data > ma20).sum(axis=1) / ma20.notna().sum(axis=1)) * 100
    pct_50 = ((data > ma50).sum(axis=1) / ma50.notna().sum(axis=1)) * 100
    pct_200 = ((data > ma200).sum(axis=1) / ma200.notna().sum(axis=1)) * 100

    idx = pct_20.dropna().tail(252).index
    valid_latest = int(data.loc[idx[-1]].notna().sum())

    return (
        pct_20.reindex(idx),
        pct_50.reindex(idx),
        pct_200.reindex(idx),
        valid_latest,
        len(tickers),
    )


def fmt_pct(series):
    val = series.iloc[-1]
    return f"{val:.1f}%" if pd.notna(val) else "N/A"


def main():
    sp500_tickers, r2000_tickers, sox_tickers = get_all_constituents()

    print("\n2. 開始下載歷史報價與運算...")
    sp500_res = calc_breadth(sp500_tickers, "S&P 500")
    r2000_res = calc_breadth(r2000_tickers, "Russell 2000")
    sox_res = calc_breadth(sox_tickers, "SOX 費半")

    print(" - 正在下載三大指數收盤價作為背景對照...")
    sp500_idx = fetch_index_close("^GSPC")
    r2000_idx = fetch_index_close("^RUT")
    sox_idx = fetch_index_close("^SOX")

    datasets = [
        ("S&P 500", sp500_res, sp500_idx),
        ("Russell 2000 羅素2000", r2000_res, r2000_idx),
        ("SOX 費城半導體", sox_res, sox_idx),
    ]

    subplot_titles = []
    for name, (p20, p50, p200, valid_n, total_n), _ in datasets:
        subplot_titles.append(
            f"<b>{name}</b> ｜ 20MA: {fmt_pct(p20)} ・ 50MA: {fmt_pct(p50)} "
            f"・ 200MA: {fmt_pct(p200)} "
            f"<span style='font-size:12px;color:#64748b;'>(有效樣本: {valid_n}/{total_n})</span>"
        )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    for i, (name, (p20, p50, p200, _, _), idx_close) in enumerate(datasets, start=1):
        aligned_idx = idx_close.reindex(p20.index).ffill()

        fig.add_trace(
            go.Scatter(
                x=aligned_idx.index,
                y=aligned_idx,
                name="指數收盤價 (右軸)",
                line=dict(color="rgba(100, 116, 139, 0.35)", width=1.6),
                hovertemplate="指數收盤: <b>%{y:,.2f}</b><extra></extra>",
                legendgroup="index_price",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=p20.index,
                y=p20,
                name="高於 20MA",
                line=dict(color="#f97316", width=2),
                hovertemplate="高於 20MA: <b>%{y:.1f}%</b><extra></extra>",
                legendgroup="20ma",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=p50.index,
                y=p50,
                name="高於 50MA",
                line=dict(color="#2563eb", width=2),
                hovertemplate="高於 50MA: <b>%{y:.1f}%</b><extra></extra>",
                legendgroup="50ma",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=p200.index,
                y=p200,
                name="高於 200MA",
                line=dict(color="#7c3aed", width=1.5, dash="dot"),
                hovertemplate="高於 200MA: <b>%{y:.1f}%</b><extra></extra>",
                legendgroup="200ma",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
            secondary_y=False,
        )

        fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", opacity=0.45, row=i, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="#94a3b8", opacity=0.4, row=i, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#22c55e", opacity=0.45, row=i, col=1)

        fig.update_yaxes(range=[0, 100], ticksuffix="%", title_text="寬度 (%)", row=i, col=1, secondary_y=False)
        fig.update_yaxes(showgrid=False, title_text="指數點位", row=i, col=1, secondary_y=True)

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        hoverformat="%Y-%m-%d (%a)",
        showticklabels=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#94a3b8",
        spikethickness=1,
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="近1個月", step="month", stepmode="backward"),
                dict(count=3, label="近3個月", step="month", stepmode="backward"),
                dict(count=6, label="近半年", step="month", stepmode="backward"),
                dict(step="all", label="全部 (1年)"),
            ],
            bgcolor="#f1f5f9",
        ),
        row=1,
        col=1,
    )

    tw_tz = timezone(timedelta(hours=8))
    updated_at = datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M CST")
    latest_trade_date = sp500_res[0].index[-1].strftime("%Y-%m-%d")

    fig.update_layout(
        height=1080,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=60, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )

    os.makedirs("public", exist_ok=True)
    chart_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", config={"responsive": True}
    )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>美股三大指數市場寬度儀表板</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
      margin: 0;
      padding: 20px;
      background: #f8fafc;
      color: #0f172a;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 1px 6px rgba(15, 23, 42, 0.08);
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 22px;
    }}
    .meta {{
      font-size: 13px;
      color: #64748b;
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>美股三大指數市場寬度儀表板（S&amp;P 500 / Russell 2000 / SOX）</h1>
    <div class="meta">
      最新美股交易日：<b>{latest_trade_date}</b> ｜ 網頁自動更新時間：{updated_at} ｜ 點擊右上角圖例可自由開關各條曲線
    </div>
    {chart_html}
  </div>
</body>
</html>
"""

    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(page_html)

    print("3. 成功產出 public/index.html！")


if __name__ == "__main__":
    main()
