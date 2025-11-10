# =========================
# Data360 (WB_WDI) Explorer — Streamlit full app
# Giữ nguyên các tính năng: bảng dữ liệu, CSV, pivot/heatmap, biểu đồ, AI insight
# Thay phần search & gọi data theo yêu cầu:
#   - Search: POST /data360/searchv2 (WB_WDI + type 'indicator')
#   - ID: full_id = WB_WDI_SP_POP_TOTL, pretty_id = SP.POP.TOTL
#   - Data: GET /data360/data?DATABASE_ID=WB_WDI&INDICATOR=<full_id>&[REF_AREA=...]
#   - Hỗ trợ chọn ALL indicators & ALL quốc gia
# =========================

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import time
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# (Tuỳ chọn) AI insight với Google Generative AI — set GOOGLE_API_KEY nếu muốn dùng
try:
    import google.generativeai as genai
except Exception:
    genai = None

# =========================
# Config
# =========================
DATA360_BASE_URL = os.environ.get("DATA360_BASE_URL", "https://api.data360.org")  # cập nhật theo môi trường của bạn
SEARCH_ENDPOINT  = "/data360/searchv2"
DATA_ENDPOINT    = "/data360/data"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

REQ_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF     = 1.6

DEFAULT_DATE_RANGE = (2004, 2024)  # chỉ phục vụ filter/biểu đồ phía client (response vẫn lấy đủ)

# =========================
# HTTP Helpers (retry + backoff)
# =========================
def _retry_sleep(attempt: int, base: float = BACKOFF) -> float:
    return min(base ** attempt, 10.0)

def http_post_json(url: str, json_body: Dict[str, Any]):
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
            time.sleep(_retry_sleep(attempt))
    raise RuntimeError(f"POST {url} failed after retries: {last_err}")

def http_get_json(url: str, params: Dict[str, Any]):
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
            time.sleep(_retry_sleep(attempt))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")

# =========================
# Search + ID helpers
# =========================
def cut_wb_id(full_id: str) -> str:
    # WB_WDI_SP_POP_TOTL -> SP_POP_TOTL
    return full_id[len("WB_WDI_"):] if full_id.startswith("WB_WDI_") else full_id

def to_pretty_id(full_id: str) -> str:
    # WB_WDI_SP_POP_TOTL -> SP.POP.TOTL
    return cut_wb_id(full_id).replace("_", ".")

@st.cache_data(show_spinner=False, ttl=600)
def search_indicators(keyword: str, top: int = 25) -> List[Dict[str, Any]]:
    """POST /data360/searchv2 — lọc WB_WDI + indicator; trả list {full_id, pretty_id, name, database_id}"""
    url = f"{DATA360_BASE_URL}{SEARCH_ENDPOINT}"
    body = {
        "count": True,
        "select": "series_description/idno, series_description/name, series_description/database_id",
        "search": keyword,
        "top": int(top),
        "filter": "series_description/database_id eq 'WB_WDI' and type eq 'indicator'"
    }
    raw = http_post_json(url, body)

    # Chuẩn hoá kết quả (tùy backend có thể là 'value' hoặc 'items')
    rows = raw.get("value") or raw.get("items") or raw
    if isinstance(rows, dict):
        rows = rows.get("value") or rows.get("items") or []

    results = []
    for row in rows:
        idno = row.get("series_description/idno")
        name = row.get("series_description/name")
        dbid = row.get("series_description/database_id")

        if idno is None and isinstance(row.get("series_description"), dict):
            sd = row["series_description"]
            idno = sd.get("idno")
            name = name or sd.get("name")
            dbid = dbid or sd.get("database_id")

        if not idno:
            continue

        results.append({
            "full_id": idno,
            "pretty_id": to_pretty_id(idno),
            "name": name or idno,
            "database_id": dbid or "WB_WDI",
        })
    return results

# =========================
# Data fetch
# =========================
@st.cache_data(show_spinner=False, ttl=600)
def fetch_data(full_indicator_id: str, ref_area: Optional[str]) -> pd.DataFrame:
    """
    GET /data360/data?DATABASE_ID=WB_WDI&INDICATOR=<full>&[REF_AREA=...]
    - Nếu ref_area None hoặc 'ALL' -> không gửi REF_AREA
    Trả DataFrame có cột chuẩn nếu tìm thấy: REF_AREA, TIME_PERIOD, VALUE, INDICATOR
    """
    url = f"{DATA360_BASE_URL}{DATA_ENDPOINT}"
    params = {"DATABASE_ID": "WB_WDI", "INDICATOR": full_indicator_id}
    if ref_area and ref_area.upper() != "ALL":
        params["REF_AREA"] = ref_area

    raw = http_get_json(url, params)

    # Chuẩn hoá: thử các dạng response thông dụng
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        rows = raw["data"]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = raw.get("value") or raw.get("items") or []

    df = pd.DataFrame(rows)

    # Đoán tên cột: (tùy hệ thống Data360 của bạn—đổi nếu cần)
    # Ưu tiên tên phổ biến:
    col_ref = next((c for c in df.columns if c.upper() in {"REF_AREA", "COUNTRY", "AREA", "LOCATION"}), None)
    col_time = next((c for c in df.columns if c.upper() in {"TIME_PERIOD", "TIME", "YEAR", "DATE"}), None)
    col_val = next((c for c in df.columns if c.upper() in {"VALUE", "OBS_VALUE", "VAL", "DATA"}), None)

    if df.empty:
        return df

    if col_ref is None or col_time is None or col_val is None:
        # Nếu không map được thì cứ trả raw + thêm indicator cho có thông tin
        df["indicator"] = full_indicator_id
        return df

    df = df.rename(columns={col_ref: "REF_AREA", col_time: "TIME_PERIOD", col_val: "VALUE"})
    df["INDICATOR"] = full_indicator_id
    return df[["REF_AREA", "TIME_PERIOD", "VALUE", "INDICATOR"]]

def fetch_many(indicator_ids: List[str], ref_area: Optional[str]) -> pd.DataFrame:
    frames = []
    progress = st.progress(0.0, text="Đang tải dữ liệu…")
    total = len(indicator_ids) if indicator_ids else 1
    for i, iid in enumerate(indicator_ids, 1):
        try:
            df_i = fetch_data(iid, ref_area)
            if not df_i.empty:
                frames.append(df_i)
        except Exception as e:
            st.warning(f"Lỗi khi tải {iid}: {e}")
        progress.progress(i/total, text=f"Đang tải {i}/{total}")
    progress.empty()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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
    if method == "Forward-fill theo quốc gia + indicator":
        return df.sort_values(["REF_AREA","INDICATOR","TIME_PERIOD"]).groupby(["REF_AREA","INDICATOR"]).ffill()
    if method == "Backward-fill theo quốc gia + indicator":
        return df.sort_values(["REF_AREA","INDICATOR","TIME_PERIOD"]).groupby(["REF_AREA","INDICATOR"]).bfill()
    return df

def filter_year_range(df: pd.DataFrame, y_from: int, y_to: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # TIME_PERIOD có thể là số hoặc chuỗi; cố gắng ép
    t = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    mask = (t >= y_from) & (t <= y_to)
    return df.loc[mask].copy()

# =========================
# UI
# =========================
st.set_page_config(page_title="Data360 — WB_WDI Explorer", layout="wide")
st.title("🔎 Data360 — WB_WDI Explorer")
st.caption("Giữ nguyên tính năng: bảng dữ liệu • CSV • pivot/heatmap • biểu đồ • AI insight.  Search/Data theo flow mới (Data360).")

# --- Search zone
with st.container():
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        keyword = st.text_input("Từ khoá indicator (ví dụ: GDP, poverty…)", value="")
    with c2:
        top_n = st.number_input("Top kết quả", 1, 200, 25, 1)
    with c3:
        search_clicked = st.button("🔍 Tìm indicator (WB_WDI)")

search_results: List[Dict[str, Any]] = []
if search_clicked:
    if not keyword.strip():
        st.warning("Vui lòng nhập từ khoá.")
    else:
        with st.spinner("Đang tìm indicator…"):
            search_results = search_indicators(keyword.strip(), int(top_n))
        if not search_results:
            st.info("Không tìm thấy indicator phù hợp.")

if search_results:
    st.subheader("Kết quả indicator")
    st.dataframe(pd.DataFrame([{
        "Indicator name": r["name"],
        "WB_ID (full)": r["full_id"],
        "WB_ID (pretty)": r["pretty_id"]
    } for r in search_results]), use_container_width=True, hide_index=True)

    st.markdown("**Chọn indicator** (hỗ trợ **ALL**)")

    options = ["ALL"] + [f'{r["name"]} — {r["pretty_id"]}' for r in search_results]
    picked = st.multiselect("Indicators", options, default=["ALL"])

    if "ALL" in picked:
        chosen_full_ids = [r["full_id"] for r in search_results]
    else:
        lookup = {f'{r["name"]} — {r["pretty_id"]}': r["full_id"] for r in search_results}
        chosen_full_ids = [lookup[x] for x in picked if x in lookup]

    st.markdown("---")

    # Quốc gia: ALL hoặc mã đơn/đa (phân tách bằng dấu phẩy -> sẽ lặp fetch từng mã)
    st.subheader("Quốc gia / Khu vực (REF_AREA)")
    st.caption("Nhập **ALL** để lấy toàn bộ; hoặc nhập 1 hay nhiều mã (VD: VNM,USA,FRA).")
    ref_area_raw = st.text_input("REF_AREA", value="ALL")

    y_from, y_to = DEFAULT_DATE_RANGE
    y_from, y_to = st.slider("Khoảng năm (lọc hiển thị, không ảnh hưởng request)", min_value=1960, max_value=2025, value=(y_from, y_to))

    na_method = st.selectbox("Xử lý N/A", ["Giữ nguyên (N/A)", "Điền 0", "Forward-fill theo quốc gia + indicator", "Backward-fill theo quốc gia + indicator"])

    tabs = st.tabs(["Dữ liệu", "Pivot & Heatmap", "Biểu đồ", "AI insight"])

    # === Tab 1: Dữ liệu ===
    with tabs[0]:
        if st.button("📥 Lấy dữ liệu"):
            if not chosen_full_ids:
                st.warning("Chọn ít nhất 1 indicator (hoặc ALL).")
                st.stop()

            ref_tokens = [x.strip() for x in ref_area_raw.split(",") if x.strip()] if ref_area_raw.strip().upper() != "ALL" else ["ALL"]

            frames = []
            with st.spinner("Đang tải dữ liệu…"):
                for ref in ref_tokens:
                    if len(chosen_full_ids) == 1:
                        frames.append(fetch_data(chosen_full_ids[0], ref))
                    else:
                        frames.append(fetch_many(chosen_full_ids, ref))

            df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()
            if df is None or df.empty:
                st.info("Không có dữ liệu.")
                st.stop()

            # Lọc theo năm phía client
            df = filter_year_range(df, y_from, y_to)

            # Xử lý N/A
            df = handle_na(df, na_method)

            st.success(f"Số dòng: {len(df)}")
            st.dataframe(df, use_container_width=True)

            # CSV
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("💾 Tải CSV", data=csv, file_name="data360_wb_wdi.csv", mime="text/csv")

            # Lưu tạm vào session state cho tab khác dùng
            st.session_state["last_df"] = df

    # === Tab 2: Pivot & Heatmap ===
    with tabs[1]:
        df: pd.DataFrame = st.session_state.get("last_df")
        if df is None or df.empty:
            st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải trước.")
        else:
            idx_cols = st.multiselect("Chọn chỉ mục (index) cho pivot", ["REF_AREA", "INDICATOR", "TIME_PERIOD"], default=["REF_AREA", "TIME_PERIOD"])
            val_agg = st.selectbox("Hàm tổng hợp", ["mean", "sum", "min", "max", "median"], index=0)

            try:
                pt = pd.pivot_table(df, index=idx_cols, values="VALUE", aggfunc=val_agg)
                st.dataframe(pt, use_container_width=True)

                # Nếu pivot thành dạng REF_AREA x TIME_PERIOD -> heatmap
                if set(idx_cols) == {"REF_AREA", "TIME_PERIOD"}:
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

    # === Tab 3: Biểu đồ ===
    with tabs[2]:
        df: pd.DataFrame = st.session_state.get("last_df")
        if df is None or df.empty:
            st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải trước.")
        else:
            # Line chart theo thời gian, phân tách theo REF_AREA/INDICATOR
            hue = st.selectbox("Tô màu theo", options=["REF_AREA", "INDICATOR"], index=0)
            try:
                fig = px.line(df.sort_values("TIME_PERIOD"), x="TIME_PERIOD", y="VALUE", color=hue, markers=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Không vẽ được biểu đồ: {e}")

    # === Tab 4: AI insight ===
    with tabs[3]:
        df: pd.DataFrame = st.session_state.get("last_df")
        if df is None or df.empty:
            st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải trước.")
        else:
            st.caption("Tóm tắt nhanh bằng AI (nếu có GOOGLE_API_KEY).")
            if genai is None or not os.environ.get("GOOGLE_API_KEY"):
                st.info("Chưa cấu hình GOOGLE_API_KEY — bỏ qua AI insight.")
            else:
                try:
                    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    # Giới hạn dữ liệu đưa vào prompt để nhanh
                    sample = df.head(100).to_dict(orient="records")
                    prompt = (
                        "Bạn là chuyên gia dữ liệu. Hãy phân tích xu hướng chính, điểm bất thường, "
                        "so sánh nhanh giữa quốc gia & chỉ số trong dữ liệu WB_WDI dưới đây. "
                        "Đề xuất 2-3 insight hành động.\n\n"
                        f"Dữ liệu mẫu (100 dòng): {sample}"
                    )
                    resp = model.generate_content(prompt)
                    st.markdown(resp.text or "_Không có phản hồi_")
                except Exception as e:
                    st.warning(f"AI insight lỗi: {e}")
