# =========================
# app.py — Streamlit + Data360 search (WB_WDI) + World Bank-style processing
# =========================

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import time
from typing import Dict, Any, Optional, List
import re
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff
import google.generativeai as genai

# ---------- Page ----------
st.set_page_config(page_title="World Bank API — Chọn theo TÊN", layout="wide", initial_sidebar_state="expanded")
st.title("Tải dữ liệu trực tiếp từ World Bank API")
st.caption("Chọn **tên** chỉ số → hệ thống tự tra **ID** (WDI) → gọi nguồn dữ liệu → hiển thị bảng/pivot/biểu đồ/CSV/AI.")

# ---------- Config ----------
WB_BASE = "https://api.worldbank.org/v2"
DATA360_BASE = "https://dataapi.worldbank.org/data360"

DEFAULT_DATE_RANGE = (2004, 2024)

HEADERS = {"User-Agent": "Streamlit-WB-Client/1.0 (contact: you@example.com)"}
DATA360_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ---------- Small utils ----------
def _to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _to_float(x, default=None):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def handle_na(df, na_method="Giữ nguyên (N/A)"):
    """Xử lý NaN trong DataFrame theo phương án được chọn."""
    if df is None or df.empty:
        return df
    if na_method == "Giữ nguyên (N/A)":
        return df
    elif na_method == "Điền giá trị gần nhất (Forward Fill)":
        return df.ffill()
    elif na_method == "Điền trung bình theo cột (Mean)":
        return df.apply(lambda x: x.fillna(x.mean()), axis=0)
    else:
        return df

# ---------- HTTP helpers ----------
def http_get_json(url: str, params: Dict[str, Any], retries: int = 4, backoff: float = 1.5):
    attempt, last_err = 0, None
    while attempt <= retries:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            last_err = e
            if e.response is not None and e.response.status_code == 429:
                ra = e.response.headers.get("Retry-After")
                sleep_s = max(backoff, int(ra)) if ra and str(ra).isdigit() else backoff * (2 ** attempt)
            else:
                sleep_s = backoff * (2 ** attempt)
            time.sleep(min(sleep_s, 12))
            attempt += 1
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
            attempt += 1
    raise last_err

def data360_request_json(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    retries: int = 4,
    backoff: float = 1.5,
):
    """Generic helper for Data360 endpoints (supports both GET/POST)."""
    attempt, last_err = 0, None
    url = f"{DATA360_BASE}{endpoint}"
    while attempt <= retries:
        try:
            resp = requests.request(
                method,
                url,
                params=params,
                json=json_payload,
                headers=DATA360_HEADERS,
                timeout=60,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            last_err = e
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                sleep_s = max(backoff, int(ra)) if ra and str(ra).isdigit() else backoff * (2 ** attempt)
            else:
                sleep_s = backoff * (2 ** attempt)
            time.sleep(min(sleep_s, 12))
            attempt += 1
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
            attempt += 1
    raise last_err

# ---------- WDI ID helpers ----------
def _format_indicator_code(raw: str) -> str:
    """Convert SP_POP_TOTL -> SP.POP.TOTL for display."""
    cleaned = (raw or "").strip("_")
    return cleaned.replace("_", ".")

def _extract_indicator_parts(full_id: str) -> tuple[str, str]:
    """
    Split WB_WDI_SP_POP_TOTL -> ("WB_WDI", "SP_POP_TOTL").
    Returns (database_id, short_indicator).
    """
    parts = (full_id or "").split("_", 2)
    if len(parts) >= 3:
        return "_".join(parts[:2]), parts[2]
    return "", full_id or ""

# ---------- Catalog (fallback) ----------
@st.cache_data(show_spinner=False, ttl=24*3600)
def wb_list_countries() -> pd.DataFrame:
    out, page = [], 1
    while True:
        js = http_get_json(f"{WB_BASE}/country", {"format":"json","per_page":400,"page":page})
        if not isinstance(js, list) or len(js) < 2:
            break
        meta, data = js
        per_page, total = _to_int(meta.get("per_page",0)), _to_int(meta.get("total",0))
        for c in data:
            if (c.get("region") or {}).get("id") != "NA":
                out.append({"code": c["id"], "name": c["name"]})
        if page * per_page >= total:
            break
        page += 1
    return pd.DataFrame(out).sort_values("name").reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_indicator_catalog() -> pd.DataFrame:
    """World Bank catalog (fallback) — returns name + wb_dot_id (e.g., NY.GDP.MKTP.CD)."""
    base = f"{WB_BASE}/indicator"
    per_page = 20000
    url = f"{base}?format=json&per_page={per_page}"
    js = http_get_json(url, {})
    items = js[1] if isinstance(js, list) and len(js) > 1 else []
    rows = []
    for it in items:
        iid = it.get("id")  # NY.GDP.MKTP.CD
        name = it.get("name") or iid
        if not iid:
            continue
        # Chỉ nhận pattern WDI chuẩn (A.B.C.D…)
        if not re.match(r"^[A-Z]{2}\.[A-Z0-9]+\.[A-Z0-9.]+$", iid):
            continue
        rows.append({"name": name, "wb_dot_id": iid})
    return pd.DataFrame(rows)

# ---------- SEARCH ----------
@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_search_indicators(keyword: str, top: int = 40) -> pd.DataFrame:
    """
    Search indicators qua Data360 (/searchv2) -> chỉ nhận idno bắt đầu bằng WB_WDI_.
    Nếu không có kết quả hợp lệ -> fallback World Bank catalog.
    """
    keyword = (keyword or "").strip()
    # --- 1) Thử Data360 ---
    try:
        payload = {
            "count": True,
            "search": keyword,
            "select": "series_description/idno, series_description/name, series_description/database_id",
            "top": max(5, top),
            "filter": "series_description/database_id eq 'WB_WDI' and type eq 'indicator'"
        }
        js = data360_request_json("POST", "/searchv2", json_payload=payload)
        values = js.get("value", []) if isinstance(js, dict) else []
        rows = []
        for item in values:
            sd = item.get("series_description") or {}
            full_id = sd.get("idno") or item.get("series_description/idno", "")
            dbid = sd.get("database_id") or item.get("series_description/database_id", "")
            if dbid != "WB_WDI" or not full_id or not full_id.startswith("WB_WDI_"):
                continue
            short_id = _extract_indicator_parts(full_id)[1]  # SP_POP_TOTL
            display_code = _format_indicator_code(short_id)  # SP.POP.TOTL
            rows.append({
                "id": full_id,                  # WB_WDI_SP_POP_TOTL
                "name": sd.get("name") or item.get("series_description/name", ""),
                "display_code": display_code,   # pretty: SP.POP.TOTL
                "wb_id": dbid
            })
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["id"]).sort_values("name").reset_index(drop=True)
    except Exception:
        pass

    # --- 2) Fallback: World Bank catalog ---
    cat = wb_indicator_catalog()
    if keyword:
        k = keyword.lower()
        cat = cat[cat["name"].str.lower().str.contains(k) | cat["wb_dot_id"].str.lower().str.contains(k)]
    cat = cat.head(top)
    # Đồng nhất schema với Data360 result
    cat = cat.assign(id=None, display_code=cat["wb_dot_id"], wb_id="WB_WDI")
    return cat.rename(columns={"wb_dot_id":"display_code"})[["id","name","display_code","wb_id"]]

# ---------- FETCH DATA (Data360 /data) ----------
@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_fetch_series(country_code: str, indicator_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Trả DF cột: Year, Country, IndicatorID, Value
    Lấy từ Data360 /data (DATABASE_ID=WB_WDI).
    Hỗ trợ cả OBS_VALUE hoặc VALUE, TIME_PERIOD có thể là 'YYYY' hoặc 'YYYY-XX'.
    """
    params = {
        "DATABASE_ID": "WB_WDI",
        "INDICATOR": indicator_id,
        "REF_AREA": country_code,
        "TIME_PERIOD": f"{year_from}:{year_to}",
    }
    js = data360_request_json("GET", "/data", params=params)

    # data360: thường có 'value' / 'data'
    values = []
    if isinstance(js, dict):
        if "value" in js and isinstance(js["value"], list):
            values = js["value"]
        elif "data" in js and isinstance(js["data"], list):
            values = js["data"]

    rows = []
    for entry in values:
        ref_area = entry.get("REF_AREA") or entry.get("REFAREA")
        period = str(entry.get("TIME_PERIOD", "") or entry.get("TIMEPERIOD", "")).strip()
        if ref_area != country_code or not period:
            continue
        year_str = period[:4] if len(period) >= 4 else period
        if not year_str.isdigit():
            continue
        # Giá trị
        obs_val = entry.get("OBS_VALUE")
        if obs_val is None:
            obs_val = entry.get("VALUE")
        obs_val = _to_float(obs_val)

        rows.append(
            {
                "Year": int(year_str),
                "Country": country_code,
                "IndicatorID": indicator_id,
                "Value": obs_val,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    out = pd.DataFrame(rows).dropna(subset=["Year"]).sort_values("Year")
    return out

def pivot_wide(df_long: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    """
    Long (Year, Country, IndicatorID, Value) -> Wide (mỗi indicator một cột).
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    df = df_long.copy()
    # Tên cột là tên chỉ số; nếu thiếu dùng IndicatorID
    df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(
        index=["Year","Country"],
        columns="IndicatorName",
        values="Value",
        aggfunc="first"
    )
    return wide.reset_index().sort_values(["Country","Year"])

# ---------------- Sidebar ----------------
st.sidebar.header("Thiết lập")

countries_df = wb_list_countries()
names = countries_df["name"].tolist()
default_idx = names.index("Viet Nam") if "Viet Nam" in names else 0
country_display = st.sidebar.selectbox(
    "Quốc gia",
    [f"{r.name} ({r.code})" for r in countries_df.itertuples()],
    index=default_idx
)
country_code = country_display.split("(")[-1].strip(")")
selected_country = country_display.split("(")[0].strip()

min_year, max_year = DEFAULT_DATE_RANGE
c1, c2 = st.sidebar.columns(2)
y_from = c1.number_input("Từ năm", min_value=1960, max_value=2100, value=min_year, step=1)
y_to   = c2.number_input("Đến năm", min_value=1960, max_value=2100, value=max_year, step=1)

st.sidebar.subheader("Tìm & chọn chỉ số (theo *tên*)")
kw = st.sidebar.text_input("Từ khoá (vd: GDP, CPI, inflation...)", value="GDP")

# Xử lý N/A
st.sidebar.subheader("Xử lý dữ liệu (Phương án xử lý N/A)")
na_method = st.sidebar.selectbox(
    "Phương án xử lý N/A (Áp dụng cho tất cả)",
    ["Giữ nguyên (N/A)", "Điền giá trị gần nhất (Forward Fill)", "Điền trung bình theo cột (Mean)"]
)

if "ind_df_cache_api" not in st.session_state:
    st.session_state["ind_df_cache_api"] = pd.DataFrame()

if st.sidebar.button("🔍 Tìm chỉ số"):
    with st.spinner("Đang tìm indicators..."):
        st.session_state["ind_df_cache_api"] = wb_search_indicators(kw, top=40)

ind_df = st.session_state["ind_df_cache_api"]
with st.sidebar.expander("Kết quả tìm thấy", expanded=False):
    if ind_df.empty:
        st.info("Nhấn **Tìm chỉ số** để tra cứu.")
    else:
        display_cols = (
            ind_df[["display_code","name","wb_id","id"]]
            .rename(columns={
                "display_code": "Indicator (WDI)",
                "name": "Tên chỉ số",
                "wb_id": "DB",
                "id": "Full ID"
            })
        )
        st.dataframe(display_cols, use_container_width=True, height=260)

# ---- Chọn indicator (có ALL) ----
indicator_names = ind_df["name"].tolist() if not ind_df.empty else []
options = (["ALL (chọn tất cả)"] + indicator_names) if indicator_names else []
default_selected = ["ALL (chọn tất cả)"] if options else []
selected_indicator_names = st.sidebar.multiselect(
    "Chọn **tên** chỉ số để lấy dữ liệu",
    options=options,
    default=default_selected
)
if "ALL (chọn tất cả)" in selected_indicator_names:
    selected_indicator_names = indicator_names

# Map name -> id (full id) để gọi /data
name_to_id = {row["name"]: row["id"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}
id_to_name = {v: k for k, v in name_to_id.items() if v is not None}

tabs = st.tabs(["📊 Dữ liệu","📈 Biểu đồ","🧮 Thống kê","📥 Tải CSV","🤖 AI"])

# == TAB 1: DỮ LIỆU ==
with tabs[0]:
    if st.button("📥 Lấy dữ liệu"):
        if not selected_indicator_names:
            st.warning("Chọn ít nhất một *tên* chỉ số.")
            st.stop()

        chosen_ids: List[str] = [name_to_id[n] for n in selected_indicator_names if n in name_to_id and name_to_id[n]]
        # Nếu kết quả từ fallback (không có full_id), báo cho người dùng
        if not chosen_ids:
            st.error("Các chỉ số đang chọn không có 'Full ID' (do lấy từ fallback). Hãy bấm Tìm chỉ số để lấy từ Data360 trước.")
            st.stop()

        all_long = []
        with st.spinner(f"Tải {len(chosen_ids)} chỉ số cho {country_code}..."):
            for ind_id in chosen_ids:
                df_fetch = wb_fetch_series(country_code, ind_id, int(y_from), int(y_to))
                if df_fetch is not None and not df_fetch.empty:
                    all_long.append(df_fetch)
                time.sleep(0.2)

        if not all_long:
            st.error("Không có dữ liệu phù hợp cho phạm vi năm/chỉ số đã chọn.")
            st.stop()

        df_long = pd.concat(all_long, ignore_index=True)
        if df_long.empty:
            st.error("Không có dữ liệu sau khi tổng hợp.")
            st.stop()

        df_wide = pivot_wide(df_long, id_to_name)
        df_wide = handle_na(df_wide, na_method)
        st.session_state["wb_df"] = df_wide

        st.success("✅ Đã tải dữ liệu.")
        st.dataframe(df_wide.set_index("Year"), use_container_width=True)

def _get_df():
    return st.session_state.get("wb_df", pd.DataFrame())

# == TAB 2: BIỂU ĐỒ ==
with tabs[1]:
    st.subheader("Biểu đồ")
    df = _get_df()
    if df.empty:
        st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải.")
    else:
        df = handle_na(df, na_method)
        cols = [c for c in df.columns if c not in ("Year", "Country")]
        choose = st.multiselect("Chọn cột vẽ", options=cols, default=cols[:min(6, len(cols))])

        if choose:
            st.plotly_chart(px.line(df, x="Year", y=choose, title="Xu hướng"), use_container_width=True)

            # Heatmap tương quan (nếu chọn >1 cột)
            if len(choose) > 1:
                df_sel = df[choose].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if df_sel.shape[1] >= 2:
                    corr = df_sel.corr().fillna(0)
                    hm = ff.create_annotated_heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale="Viridis",
                        annotation_text=corr.round(2).values,
                        showscale=True,
                    )
                    st.plotly_chart(hm, use_container_width=True)

# == TAB 3: THỐNG KÊ ==
with tabs[2]:
    st.subheader("Thống kê mô tả")
    df = _get_df()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        df = handle_na(df, na_method)
        cols = [c for c in df.columns if c not in ("Year", "Country")]
        if not cols:
            st.info("Không có cột số để thống kê.")
        else:
            stats = df[cols].apply(pd.to_numeric, errors="coerce").describe().T
            stats["CV"] = (stats["std"]/stats["mean"]).abs()
            st.dataframe(
                stats[["mean","std","min","50%","max","CV"]]
                .rename(columns={"mean":"Mean","std":"Std","50%":"Median"}),
                use_container_width=True
            )

# == TAB 4: TẢI CSV ==
with tabs[3]:
    st.subheader("Tải CSV")
    df = _get_df()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.download_button(
            "📥 Tải CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"wb_{country_code}_{y_from}_{y_to}.csv",
            mime="text/csv"
        )

# == TAB 5: AI PHÂN TÍCH VÀ TƯ VẤN ==
with tabs[4]:
    st.header("AI phân tích và tư vấn")
    selected_start_year = int(y_from)
    selected_end_year = int(y_to)
    df_processed_sidebar = _get_df()
    target_audience = "Ngân hàng Agribank"
    st.subheader(f"Đối tượng tư vấn: {target_audience}")

    def generate_ai_analysis(data_df: pd.DataFrame, country: str, audience: str):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel('gemini-2.5-pro')
            data_string = data_df.to_csv(index=False)

            prompt_template = f"""
            Bạn là một chuyên gia phân tích kinh tế vĩ mô hàng đầu, đang chuẩn bị một báo cáo tư vấn.
            Dưới đây là bộ dữ liệu kinh tế vĩ mô của **{country}** từ năm {selected_start_year} đến {selected_end_year}:

            {data_string}

            Hãy trình bày ngắn gọn, có số liệu minh họa, kết thúc với 3 khuyến nghị hành động cụ thể cho **{audience}**.
            """
            with st.spinner(f"AI đang phân tích {country} và tạo báo cáo cho {audience}..."):
                response = model.generate_content(prompt_template)
                return response.text

        except Exception as e:
            msg = str(e)
            if "GEMINI_API_KEY" in msg and "not found" in msg.lower():
                st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thiết lập trong file .streamlit/secrets.toml")
            elif "API key is invalid" in msg:
                st.error("Lỗi: GEMINI_API_KEY không hợp lệ. Vui lòng kiểm tra lại trong file .streamlit/secrets.toml")
            else:
                st.error(f"Đã xảy ra lỗi khi gọi AI: {e}")
            return None

    if st.button(f"🚀 Sinh AI phân tích và tư vấn cho {target_audience}"):
        if df_processed_sidebar.empty:
            st.error("Không có dữ liệu để phân tích. Vui lòng chọn chỉ số ở các tab trước và nhấn 'Lấy dữ liệu'.")
        else:
            ai_report = generate_ai_analysis(df_processed_sidebar, selected_country, target_audience)
            if ai_report:
                st.markdown(ai_report)
