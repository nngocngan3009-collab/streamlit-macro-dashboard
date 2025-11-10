# =========================
# Data360 → WB API Explorer (Streamlit, WB_WDI only)
# =========================

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import time
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff

# (Tuỳ chọn) AI insight qua Google Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# =========================
# Config
# =========================
DATA360_BASE_URL = os.environ.get("DATA360_BASE_URL", "https://api.data360.org")
SEARCH_ENDPOINT  = "/data360/searchv2"

WB_BASE = "https://api.worldbank.org/v2"

HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
REQ_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF     = 1.6

DEFAULT_TOP = 25
DEFAULT_YEAR_RANGE = (2000, 2024)

# =========================
# HTTP helpers (retry + backoff)
# =========================
def _sleep(attempt: int, base: float = BACKOFF) -> float:
    return min(base ** attempt, 10.0)

def http_post_json(url: str, json_body: Dict[str, Any]) -> Any:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=json_body, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(attempt))
    raise RuntimeError(f"POST {url} failed after retries: {last_err}")

def http_get(url: str, params: Dict[str, Any]) -> requests.Response:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(_sleep(attempt))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")

# =========================
# ID helpers
# =========================
def cut_wb_id(full_id: str) -> str:
    """WB_WDI_SP_POP_TOTL -> SP_POP_TOTL"""
    return full_id[len("WB_WDI_"):] if full_id.startswith("WB_WDI_") else full_id

def pretty_id(full_id: str) -> str:
    """WB_WDI_SP_POP_TOTL -> SP.POP.TOTL"""
    return cut_wb_id(full_id).replace("_", ".")

# =========================
# API wrappers
# =========================
@st.cache_data(show_spinner=False, ttl=1200)
def d360_search_indicators(keyword: str, top: int = DEFAULT_TOP) -> pd.DataFrame:
    """
    POST /data360/searchv2 — lọc WB_WDI + indicator
    Trả DF: name, full_id (WB_WDI_*), wb_id (SP_POP_TOTL), pretty_id (SP.POP.TOTL)
    """
    body = {
        "count": True,
        "select": "series_description/idno, series_description/name, series_description/database_id",
        "search": keyword,
        "top": int(top),
        "filter": "series_description/database_id eq 'WB_WDI' and type eq 'indicator'"
    }
    raw = http_post_json(f"{DATA360_BASE_URL}{SEARCH_ENDPOINT}", body)

    rows = raw.get("value") or raw.get("items") or raw
    if isinstance(rows, dict):
        rows = rows.get("value") or rows.get("items") or []

    items: List[Dict[str, Any]] = []
    for r in rows:
        sd = r.get("series_description") if isinstance(r.get("series_description"), dict) else None
        idno = r.get("series_description/idno") or (sd.get("idno") if sd else None)
        name = r.get("series_description/name") or (sd.get("name") if sd else None)
        if not idno:
            continue
        items.append({
            "name": name or idno,
            "full_id": idno,                  # WB_WDI_SP_POP_TOTL
            "wb_id": cut_wb_id(idno),         # SP_POP_TOTL
            "pretty_id": pretty_id(idno),     # SP.POP.TOTL
        })
    return pd.DataFrame(items)

@st.cache_data(show_spinner=False, ttl=1200)
def wb_fetch_series(wb_dot_id: str, ref_area: Optional[str]) -> pd.DataFrame:
    """
    World Bank v2: /v2/country/{REF_AREA}/indicator/{WB_ID}?format=json&per_page=20000
    - wb_dot_id: SP.POP.TOTL, NY.GDP.MKTP.CD, ...
    - ref_area: ALL -> 'all', hoặc mã (VN, USA, VNM, …)
    """
    country_seg = "all" if not ref_area or ref_area.strip().upper() == "ALL" else ref_area.strip()
    url = f"{WB_BASE}/country/{country_seg}/indicator/{wb_dot_id}"
    params = {"format": "json", "per_page": 20000}
    r = http_get(url, params)
    payload = r.json()

    items = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        ref = (it.get("countryiso3code")
               or (it.get("country") or {}).get("id")
               or (it.get("country") or {}).get("value"))
        rows.append({
            "REF_AREA": ref,
            "TIME_PERIOD": it.get("date"),
            "VALUE": it.get("value"),
        })
    df = pd.DataFrame(rows)
    return df

def fetch_data_by_full(full_indicator_id: str, ref_area: Optional[str]) -> pd.DataFrame:
    """Nhận full_id (WB_WDI_SP_POP_TOTL) → chuyển sang WB dot id → gọi WB v2"""
    wb_dot = pretty_id(full_indicator_id)
    df = wb_fetch_series(wb_dot, ref_area)
    if df is not None and not df.empty:
        df["INDICATOR"] = full_indicator_id
    return df

def fetch_many(full_ids: List[str], ref_area: Optional[str]) -> pd.DataFrame:
    frames = []
    progress = st.progress(0.0, text="Đang tải dữ liệu…")
    n = len(full_ids) if full_ids else 1
    for i, fid in enumerate(full_ids, 1):
        try:
            frames.append(fetch_data_by_full(fid, ref_area))
        except Exception as e:
            st.warning(f"Lỗi khi lấy {pretty_id(fid)}: {e}")
        progress.progress(i/n, text=f"Đang tải {i}/{n}")
    progress.empty()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# =========================
# Data utils
# =========================
def handle_na(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if method == "Giữ nguyên (N/A)":
        return df
    if method == "Điền 0":
        return df.fillna(0)
    if method == "Forward-fill theo quốc gia + indicator":
        return (df.sort_values(["REF_AREA","INDICATOR","TIME_PERIOD"])
                  .groupby(["REF_AREA","INDICATOR"])
                  .ffill())
    if method == "Backward-fill theo quốc gia + indicator":
        return (df.sort_values(["REF_AREA","INDICATOR","TIME_PERIOD"])
                  .groupby(["REF_AREA","INDICATOR"])
                  .bfill())
    return df

def filter_year(df: pd.DataFrame, y_from: int, y_to: int) -> pd.DataFrame:
    if df is None or df.empty or "TIME_PERIOD" not in df.columns:
        return df
    t = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    return df.loc[(t >= y_from) & (t <= y_to)].copy()

# =========================
# UI — sidebar & tabs
# =========================
st.set_page_config(page_title="Data360 → World Bank Explorer", layout="wide")
st.title("🔎 Data360 → World Bank (WB_WDI)")
st.caption("Search indicator từ Data360 (WB_WDI) → gọi World Bank v2 để lấy dữ liệu. Hỗ trợ ALL indicators & ALL quốc gia.")

# Sidebar — Search
with st.sidebar:
    st.header("Tìm chỉ số (WB_WDI)")
    kw = st.text_input("Từ khoá (vd: GDP, poverty…)", value="")
    top_n = st.number_input("Top kết quả", 1, 200, DEFAULT_TOP, 1)
    if st.button("🔍 Tìm indicator (Data360)"):
        if not kw.strip():
            st.warning("Nhập từ khoá trước khi tìm.")
        else:
            with st.spinner("Đang tìm…"):
                st.session_state["ind_df_cache"] = d360_search_indicators(kw.strip(), int(top_n))

    ind_df = st.session_state.get("ind_df_cache", pd.DataFrame())
    st.write("Kết quả")
    if ind_df.empty:
        st.info("Nhấn **Tìm indicator** để tra cứu.")
    else:
        st.dataframe(ind_df[["name","pretty_id","full_id"]], height=240, use_container_width=True)

    # ALL indicators
    indicator_options = (["ALL (chọn tất cả)"] + ind_df["name"].tolist()) if not ind_df.empty else []
    default_ind_opts = ["ALL (chọn tất cả)"] if indicator_options else []
    picked_names = st.multiselect("Chọn indicator", options=indicator_options, default=default_ind_opts)

    if "ALL (chọn tất cả)" in picked_names and not ind_df.empty:
        picked_names = ind_df["name"].tolist()

    # map name → full_id để gọi data
    name_to_full = {row["name"]: row["full_id"] for _, row in ind_df.iterrows()} if not ind_df.empty else {}
    chosen_full_ids = [name_to_full[n] for n in picked_names if n in name_to_full]

    st.markdown("---")
    st.header("Quốc gia / Vùng")
    st.caption("Nhập **ALL** để lấy tất cả, hoặc nhập 1–n mã (VD: VN,USA,FRA).")
    ref_area_raw = st.text_input("REF_AREA", value="ALL")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dữ liệu", "Pivot & Heatmap", "Biểu đồ", "💾 CSV", "🤖 AI insight"])

with tab1:
    st.subheader("Lấy dữ liệu")
    y_from, y_to = st.slider("Khoảng năm (lọc hiển thị)", 1960, 2025, DEFAULT_YEAR_RANGE)
    na_method = st.selectbox("Xử lý N/A", ["Giữ nguyên (N/A)", "Điền 0",
                                           "Forward-fill theo quốc gia + indicator",
                                           "Backward-fill theo quốc gia + indicator"])
    if st.button("📥 Tải dữ liệu"):
        if not chosen_full_ids:
            st.warning("Chọn ít nhất 1 indicator (hoặc ALL).")
            st.stop()
        # Chuẩn hóa danh sách REF_AREA
        if ref_area_raw.strip().upper() == "ALL":
            ref_list = ["ALL"]
        else:
            ref_list = [x.strip() for x in ref_area_raw.split(",") if x.strip()]

        frames = []
        with st.spinner("Đang tải dữ liệu…"):
            for ref in ref_list:
                if len(chosen_full_ids) == 1:
                    frames.append(fetch_data_by_full(chosen_full_ids[0], ref))
                else:
                    frames.append(fetch_many(chosen_full_ids, ref))
        df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()

        if df.empty:
            st.info("Không có dữ liệu.")
        else:
            df = filter_year(df, y_from, y_to)
            df = handle_na(df, na_method)
            st.success(f"Số dòng: {len(df)}")
            st.dataframe(df, use_container_width=True)
            st.session_state["last_df"] = df

with tab2:
    st.subheader("Pivot & Heatmap")
    df = st.session_state.get("last_df")
    if df is None or df.empty:
        st.info("Chưa có dữ liệu — hãy tải ở tab **Dữ liệu**.")
    else:
        idx_cols = st.multiselect("Chọn index cho pivot", ["REF_AREA","INDICATOR","TIME_PERIOD"], default=["REF_AREA","TIME_PERIOD"])
        agg = st.selectbox("Hàm tổng hợp", ["mean","sum","min","max","median"], index=0)
        try:
            pt = pd.pivot_table(df, index=idx_cols, values="VALUE", aggfunc=agg)
            st.dataframe(pt, use_container_width=True)
            # Nếu pivot đúng dạng REF_AREA x TIME_PERIOD -> heatmap
            if set(idx_cols) == {"REF_AREA","TIME_PERIOD"}:
                mat = pt.reset_index().pivot(index="REF_AREA", columns="TIME_PERIOD", values="VALUE")
                fig = ff.create_annotated_heatmap(
                    z=np.array(mat.values, dtype=float),
                    x=[str(x) for x in mat.columns],
                    y=list(mat.index),
                    showscale=True
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Không tạo được pivot: {e}")

with tab3:
    st.subheader("Biểu đồ")
    df = st.session_state.get("last_df")
    if df is None or df.empty:
        st.info("Chưa có dữ liệu — hãy tải ở tab **Dữ liệu**.")
    else:
        hue = st.selectbox("Tô màu theo", ["REF_AREA","INDICATOR"], index=0)
        try:
            fig = px.line(df.sort_values("TIME_PERIOD"), x="TIME_PERIOD", y="VALUE", color=hue, markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Không vẽ được biểu đồ: {e}")

with tab4:
    st.subheader("Tải CSV")
    df = st.session_state.get("last_df")
    if df is None or df.empty:
        st.info("Chưa có dữ liệu — hãy tải ở tab **Dữ liệu**.")
    else:
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 Download CSV", data=csv, file_name="wb_wdi_data.csv", mime="text/csv")

with tab5:
    st.subheader("AI insight (tuỳ chọn)")
    df = st.session_state.get("last_df")
    if df is None or df.empty:
        st.info("Chưa có dữ liệu — hãy tải ở tab **Dữ liệu**.")
    else:
        if genai is None or not os.environ.get("GOOGLE_API_KEY"):
            st.info("Chưa cấu hình GOOGLE_API_KEY nên bỏ qua AI insight.")
        else:
            try:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                model = genai.GenerativeModel("gemini-1.5-flash")
                sample = df.head(200).to_dict(orient="records")
                prompt = (
                    "Bạn là chuyên gia dữ liệu kinh tế. Hãy tóm tắt xu hướng chính, điểm bất thường, "
                    "và gợi ý 2–3 insight hành động dựa trên dữ liệu WB_WDI sau. "
                    "Trả lời ngắn gọn, gạch đầu dòng.\n\n"
                    f"Dữ liệu mẫu (<=200 dòng): {sample}"
                )
                resp = model.generate_content(prompt)
                st.markdown(resp.text or "_Không có phản hồi_")
            except Exception as e:
                st.warning(f"AI lỗi: {e}")
