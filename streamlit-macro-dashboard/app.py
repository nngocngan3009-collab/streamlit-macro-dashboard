import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import re
import time
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff

# =========================
# Config
# =========================
WB_BASE = "https://api.worldbank.org/v2"
HEADERS = {"User-Agent": "Streamlit-WB-Client/1.0 (contact: you@example.com)",
           "Accept": "application/json"}
REQ_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF     = 1.6
DEFAULT_DATE_RANGE = (2000, 2024)

# =========================
# Helpers (retry)
# =========================

def _sleep(attempt: int, base: float = BACKOFF) -> float:
    return min(base ** attempt, 12.0)


def http_get_json(url: str, params: Dict[str, Any]) -> Any:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(attempt))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")

# =========================
# Indicator utilities
# =========================
_VALID_WB_ID = re.compile(r"^[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)+$")


def is_valid_wb_id(candidate: str) -> bool:
    if not isinstance(candidate, str):
        return False
    c = candidate.strip()
    return bool(_VALID_WB_ID.match(c))


@st.cache_data(show_spinner=False, ttl=24*3600)
def wb_search_indicators(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    results, page = [], 1
    key = (keyword or "").strip().lower()
    while page <= max_pages:
        js = http_get_json(f"{WB_BASE}/indicator", {"format":"json","per_page":5000,"page":page})
        if not isinstance(js, list) or len(js) < 2:
            break
        meta, data = js
        per_page = int((meta or {}).get("per_page", 0) or 0)
        total    = int((meta or {}).get("total", 0) or 0)
        for it in (data or []):
            _id, _name = it.get("id", ""), it.get("name", "")
            _source = (it.get("source", {}) or {}).get("value", "")
            if key and (key not in _name.lower() and key not in _id.lower()):
                continue
            if not is_valid_wb_id(_id):
                continue
            results.append({
                "id": _id,
                "name": _name,
                "unit": it.get("unit", ""),
                "source": _source
            })
        if page * per_page >= total or per_page == 0:
            break
        page += 1
    df = pd.DataFrame(results).drop_duplicates(subset=["id"]).sort_values("name").reset_index(drop=True)
    return df

# =========================
# Fetch series
# =========================
@st.cache_data(show_spinner=False, ttl=1200)
def wb_fetch_series(country_code: str, indicator_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    js = http_get_json(
        f"{WB_BASE}/country/{country_code}/indicator/{indicator_id}",
        {"format": "json", "per_page": 20000, "date": f"{int(year_from)}:{int(year_to)}"}
    )

    if not isinstance(js, list) or len(js) < 2:
        return pd.DataFrame(columns=["Year", "Country", "IndicatorID", "Value"])
    if isinstance(js[0], dict) and js[0].get("message"):
        return pd.DataFrame(columns=["Year", "Country", "IndicatorID", "Value"])

    _, data = js
    rows = []
    for d in (data or []):
        year_raw = str(d.get("date", ""))
        year = int(year_raw) if year_raw.isdigit() else None
        rows.append({
            "Year": year,
            "Country": (d.get("country") or {}).get("value", country_code),
            "IndicatorID": (d.get("indicator") or {}).get("id", indicator_id),
            "Value": d.get("value", None)
        })
    out = pd.DataFrame(rows).dropna(subset=["Year"]) if rows else pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    return out.sort_values(["Country","IndicatorID","Year"]) if not out.empty else out


def pivot_wide(df_long: pd.DataFrame, use_friendly_name: bool, id_to_name: Dict[str, str]) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    key_col = "IndicatorName" if use_friendly_name else "IndicatorID"
    df = df_long.copy()
    if use_friendly_name:
        df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(index=["Year","Country"], columns=key_col, values="Value", aggfunc="first")
    wide = wide.reset_index().sort_values(["Country","Year"])
    wide = wide.rename(columns={"Year": "Năm"})
    return wide

# =========================
# Data utilities
# =========================

def handle_na(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if method == "Giữ nguyên (N/A)":
        return df
    if method == "Điền 0":
        return df.fillna(0)
    if method == "Forward-fill theo Country + cột dữ liệu":
        cols = [c for c in df.columns if c not in ("Năm", "Country")]
        return (df.sort_values(["Country","Năm"]) \
                  .groupby("Country")[cols] \
                  .ffill() \
                  .reindex(df.index) \
                  .pipe(lambda d: df.assign(**{c: d[c] for c in cols})))
    if method == "Backward-fill theo Country + cột dữ liệu":
        cols = [c for c in df.columns if c not in ("Năm", "Country")]
        return (df.sort_values(["Country","Năm"]) \
                  .groupby("Country")[cols] \
                  .bfill() \
                  .reindex(df.index) \
                  .pipe(lambda d: df.assign(**{c: d[c] for c in cols})))
    return df

# =========================
# UI
# =========================

st.set_page_config(page_title="World Bank WDI — Sửa python7", layout="wide")
st.title("Công cụ tổng hợp và phân tích dữ liệu vĩ mô kết hợp AI")
st.caption("Tìm indicator (WDI, lọc ID hợp lệ) → Lấy dữ liệu qua API v2 → Bảng rộng: Năm, Country, chỉ số…")

# ===== Sidebar: Tool tìm indicator, chọn năm, Xử lý N/A, Quốc gia =====
with st.sidebar:
    st.header("🔧 Công cụ")
    # Quốc gia
    country_list = {
        "Vietnam (VN)": "VN", "United States (US)": "US", "China (CN)": "CN", "India (IN)": "IN"
        # Bạn có thể thêm các quốc gia khác vào đây
    }
    country_raw = st.selectbox("Chọn quốc gia", options=list(country_list.keys()), index=0)

    # Tìm indicator
    st.subheader("Tìm chỉ số (WDI)")
    kw = st.text_input("Từ khoá", value="GDP")
    top_n = st.number_input("Top", 1, 500, 50, 1)
    do_search = st.button("🔍 Tìm indicator")

    if do_search:
        if not kw.strip():
            st.warning("Nhập từ khoá trước khi tìm.")
        else:
            with st.spinner("Đang tìm indicators (WDI)…"):
                df_ind = wb_search_indicators(kw.strip(), max_pages=3)
                if top_n:
                    df_ind = df_ind.head(int(top_n))
                st.session_state["ind_search_df"] = df_ind

    # Khoảng năm + xử lý NA
    y_from, y_to = st.slider("Khoảng năm", 1995, 2025, DEFAULT_DATE_RANGE)
    na_method = st.selectbox(
        "Xử lý N/A",
        [
            "Giữ nguyên (N/A)",
            "Điền 0",
            "Forward-fill theo Country + cột dữ liệu",
            "Backward-fill theo Country + cột dữ liệu",
        ],
        index=0,
    )

# ===== Main area: Tabs riêng biệt =====
TAB_TITLES = ["📊 Dữ liệu", "📈 Biểu đồ", "🧮 Thống kê", "📥 Xuất dữ liệu", "🤖 AI"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_TITLES)

# Tải kết quả tìm kiếm để chọn indicator
ind_df = st.session_state.get("ind_search_df", pd.DataFrame())
name_to_id = {row["name"]: row["id"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}
id_to_name = {v: k for k, v in name_to_id.items()}
indicator_names = ind_df["name"].tolist() if not ind_df.empty else []

with tab1:
    st.subheader("Chọn chỉ số từ kết quả tìm kiếm (WDI)")
    if ind_df.empty:
        st.info("Hãy dùng thanh bên trái để *Tìm indicator*. Chỉ số hiển thị là từ WDI và đã lọc ID sai định dạng.")
    else:
        st.dataframe(ind_df[["id","name","unit","source"]], height=220, use_container_width=True)

    # Cho phép chọn các chỉ số
    selected_indicator_names = st.multiselect(
        "Chọn chỉ số theo TÊN (sẽ tự lắp ID vào API)",
        options=indicator_names,
        default=indicator_names[:1] if indicator_names else []
    )

    # Nút tải dữ liệu
    load_clicked = st.button("📥 Tải dữ liệu")

# Phần còn lại vẫn như cũ
