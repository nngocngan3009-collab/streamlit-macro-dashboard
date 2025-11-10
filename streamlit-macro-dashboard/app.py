# =========================
# app.py — Streamlit + Data360 search (WB_WDI) + Data360 data + Full Tabs
# =========================

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import re
import time
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff

# (Tuỳ chọn) AI qua Google Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------- Page ----------
st.set_page_config(page_title="World Bank — Data360 (WB_WDI)", layout="wide", initial_sidebar_state="expanded")
st.title("Tải dữ liệu trực tiếp từ World Bank (qua Data360)")
st.caption("Chọn **tên** chỉ số → hệ thống tự tìm **ID** (WDI) → gọi Data360 **/data** → hiển thị bảng/pivot/biểu đồ/CSV/AI.")

# ---------- Config ----------
WB_BASE = "https://api.worldbank.org/v2"            # chỉ dùng cho fallback catalog
DATA360_BASE = "https://dataapi.worldbank.org/data360"  # đổi nếu môi trường khác

DEFAULT_DATE_RANGE = (2004, 2024)

UA = "Streamlit-WB-Client/1.0 (contact: you@example.com)"
HEADERS = {"User-Agent": UA}
DATA360_HEADERS = {"User-Agent": UA, "Accept": "application/json", "Content-Type": "application/json"}

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
def http_get_json(url: str, params: Dict[str, Any], headers=None, retries: int = 4, backoff: float = 1.5):
    attempt, last_err = 0, None
    headers = headers or HEADERS
    while attempt <= retries:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
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

def data360_request_json(method: str, endpoint: str, *, params: Dict[str, Any] | None = None,
                         json_payload: Dict[str, Any] | None = None, retries: int = 4, backoff: float = 1.5):
    attempt, last_err = 0, None
    url = f"{DATA360_BASE}{endpoint}"
    while attempt <= retries:
        try:
            resp = requests.request(method, url, params=params, json=json_payload,
                                    headers=DATA360_HEADERS, timeout=60)
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
    cleaned = (raw or "").strip("_")
    return cleaned.replace("_", ".")

def _extract_indicator_parts(full_id: str) -> tuple[str, str]:
    parts = (full_id or "").split("_", 2)
    if len(parts) >= 3:
        return "_".join(parts[:2]), parts[2]
    return "", full_id or ""

# ---------- Catalog ----------
@st.cache_data(show_spinner=False, ttl=24*3600)
def wb_list_countries() -> pd.DataFrame:
    out, page = [], 1
    while True:
        js = http_get_json(f"{WB_BASE}/country", {"format": "json", "per_page": 400, "page": page})
        if not isinstance(js, list) or len(js) < 2:
            break
        meta, data = js
        per_page, total = _to_int(meta.get("per_page", 0)), _to_int(meta.get("total", 0))
        for c in data:
            if (c.get("region") or {}).get("id") != "NA":
                out.append({"code": c["id"], "name": c["name"]})
        if page * per_page >= total:
            break
        page += 1
    return pd.DataFrame(out).sort_values("name").reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_indicator_catalog() -> pd.DataFrame:
    base = f"{WB_BASE}/indicator"
    per_page = 20000
    js = http_get_json(f"{base}?format=json&per_page={per_page}", {})
    items = js[1] if isinstance(js, list) and len(js) > 1 else []
    rows = []
    for it in items:
        iid = it.get("id")  # NY.GDP.MKTP.CD
        name = it.get("name") or iid
        if iid and re.match(r"^[A-Z]{2}\.[A-Z0-9]+\.[A-Z0-9.]+$", iid):
            rows.append({"name": name, "wb_dot_id": iid})
    return pd.DataFrame(rows)

# ---------- SEARCH ----------
@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_search_indicators(keyword: str, top: int = 40) -> pd.DataFrame:
    """
    Search indicators qua Data360 (/searchv2) -> chỉ nhận idno bắt đầu WB_WDI_.
    Nếu không có kết quả hợp lệ -> fallback World Bank catalog.
    Trả DF: id (full_id), name, display_code (pretty), wb_id
    """
    keyword = (keyword or "").strip()
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
            short_id = _extract_indicator_parts(full_id)[1]        # SP_POP_TOTL
            display_code = _format_indicator_code(short_id)        # SP.POP.TOTL
            rows.append({
                "id": full_id,                 # WB_WDI_SP_POP_TOTL
                "name": sd.get("name") or item.get("series_description/name", ""),
                "display_code": display_code,  # pretty: SP.POP.TOTL
                "wb_id": dbid
            })
        if rows:
            return (pd.DataFrame(rows)
                    .drop_duplicates(subset=["id"])
                    .sort_values("name")
                    .reset_index(drop=True))
    except Exception:
        pass

    # Fallback: WB catalog
    cat = wb_indicator_catalog()
    if keyword:
        k = keyword.lower()
        cat = cat[cat["name"].str.lower().str.contains(k) | cat["wb_dot_id"].str.lower().str.contains(k)]
    cat = cat.head(top)
    # Đồng nhất schema với kết quả Data360
    cat = cat.assign(id=None, display_code=cat["wb_dot_id"], wb_id="WB_WDI")
    return cat.rename(columns={"wb_dot_id": "display_code"})[["id", "name", "display_code", "wb_id"]]

# ---------- FETCH DATA (Data360 /data) ----------
@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_fetch_series(country_code: str, indicator_full_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    """
    GET /data?DATABASE_ID=WB_WDI&INDICATOR=<full>&REF_AREA=<country>&TIME_PERIOD=YYYY:YYYY
    Chuẩn hoá output -> Year, Country, IndicatorID, Value
    """
    params = {
        "DATABASE_ID": "WB_WDI",
        "INDICATOR": indicator_full_id,
        "REF_AREA": country_code,
        "TIME_PERIOD": f"{year_from}:{year_to}",
    }
    js = data360_request_json("GET", "/data", params=params)

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
        val = entry.get("OBS_VALUE")
        if val is None:
            val = entry.get("VALUE")
        rows.append(
            {
                "Year": int(year_str),
                "Country": country_code,
                "IndicatorID": indicator_full_id,
                "Value": _to_float(val),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Year", "Country", "IndicatorID", "Value"])
    return pd.DataFrame(rows).dropna(subset=["Year"]).sort_values("Year")

def pivot_wide(df_long: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    """Long (Year, Country, IndicatorID, Value) -> Wide (mỗi indicator 1 cột)."""
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    df = df_long.copy()
    df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(
        index=["Year", "Country"],
        columns="IndicatorName",
        values="Value",
        aggfunc="first"
    )
    return wide.reset_index().sort_values(["Country", "Year"])

# ---------------- Sidebar ----------------
st.sidebar.header("Thiết lập")

# Quốc gia: ALL hoặc nhập/hoặc chọn danh sách
st.sidebar.subheader("Quốc gia (REF_AREA)")
all_countries = st.sidebar.checkbox("ALL quốc gia", value=False)
countries_df = wb_list_countries()
names = countries_df["name"].tolist()
default_idx = names.index("Viet Nam") if "Viet Nam" in names else 0
country_select = st.sidebar.selectbox(
    "Chọn nhanh 1 quốc gia",
    [f"{r.name} ({r.code})" for r in countries_df.itertuples()],
    index=default_idx
)
manual_codes = st.sidebar.text_input("Hoặc nhập mã (1-n, cách nhau dấu phẩy)", value="")

def resolve_country_list() -> List[str]:
    if all_countries:
        return ["ALL"]
    manual = [x.strip() for x in manual_codes.split(",") if x.strip()]
    if manual:
        return manual
    # lấy từ selectbox
    return [country_select.split("(")[-1].strip(")")]

country_list = resolve_country_list()

min_year, max_year = DEFAULT_DATE_RANGE
c1, c2 = st.sidebar.columns(2)
y_from = c1.number_input("Từ năm", min_value=1960, max_value=2100, value=min_year, step=1)
y_to   = c2.number_input("Đến năm", min_value=1960, max_value=2100, value=max_year, step=1)

st.sidebar.subheader("Tìm & chọn chỉ số (theo *tên*)")
kw = st.sidebar.text_input("Từ khoá (vd: GDP, CPI, inflation...)", value="GDP")

# Xử lý N/A
st.sidebar.subheader("Xử lý dữ liệu (N/A)")
na_method = st.sidebar.selectbox(
    "Phương án xử lý",
    ["Giữ nguyên (N/A)", "Điền giá trị gần nhất (Forward Fill)", "Điền trung bình theo cột (Mean)"]
)

if "ind_df_cache_api" not in st.session_state:
    st.session_state["ind_df_cache_api"] = pd.DataFrame()

if st.sidebar.button("🔍 Tìm chỉ số"):
    with st.spinner("Đang tìm indicators..."):
        st.session_state["ind_df_cache_api"] = wb_search_indicators(kw, top=60)

ind_df = st.session_state["ind_df_cache_api"]

with st.sidebar.expander("Kết quả tìm thấy", expanded=False):
    if ind_df.empty:
        st.info("Nhấn **Tìm chỉ số** để tra cứu.")
    else:
        # Hiển thị an toàn: chỉ dùng cột tồn tại + loại trùng tên
        cols_map = {"display_code": "Indicator (WDI)", "name": "Tên chỉ số", "wb_id": "DB", "id": "Full ID"}
        available = [c for c in cols_map if c in ind_df.columns]
        display_cols = ind_df[available].rename(columns={c: cols_map[c] for c in available})
        display_cols = display_cols.loc[:, ~display_cols.columns.duplicated()].copy()
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

# Map name -> id / pretty để gọi /data (nếu thiếu id thì suy ra từ pretty)
name_to_id = {row["name"]: row["id"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}
name_to_pretty = {row["name"]: row["display_code"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}

# Tabs
tabs = st.tabs(["📊 Dữ liệu", "📈 Biểu đồ", "🧮 Thống kê", "📥 Tải CSV", "🤖 AI"])

# == TAB 1: DỮ LIỆU ==
with tabs[0]:
    if st.button("📥 Lấy dữ liệu"):
        if not selected_indicator_names:
            st.warning("Chọn ít nhất một *tên* chỉ số.")
            st.stop()

        # Chuẩn hoá danh sách indicator full_id
        chosen_full_ids: List[str] = []
        id_to_name: Dict[str, str] = {}
        for n in selected_indicator_names:
            fid = name_to_id.get(n)
            if not fid:
                # fallback: suy ra từ pretty_id (vd NY.GDP.MKTP.CD -> WB_WDI_NY_GDP_MKTP_CD)
                pretty = name_to_pretty.get(n)
                if pretty:
                    fid = "WB_WDI_" + pretty.replace(".", "_")
            if fid:
                chosen_full_ids.append(fid)
                id_to_name[fid] = n

        if not chosen_full_ids:
            st.error("Không xác định được ID chỉ tiêu để gọi /data. Hãy thử tìm lại bằng Data360.")
            st.stop()

        all_long = []
        with st.spinner(f"Tải {len(chosen_full_ids)} chỉ tiêu cho {len(country_list)} quốc gia..."):
            for country_code in country_list:
                for ind_id in chosen_full_ids:
                    df_fetch = wb_fetch_series(country_code, ind_id, int(y_from), int(y_to))
                    if df_fetch is not None and not df_fetch.empty:
                        all_long.append(df_fetch)
                    time.sleep(0.05)

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
        st.dataframe(df_wide.set_index(["Country", "Year"]), use_container_width=True)

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
        group = st.selectbox("Nhóm theo", ["Country", "Không (gộp)"], index=0)
        if choose:
            if group == "Country":
                fig = px.line(df, x="Year", y=choose, color="Country", markers=True)
            else:
                fig = px.line(df, x="Year", y=choose, markers=True)
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap tương quan (nếu chọn >1 cột, chỉ tính trên các cột số)
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
            stats["CV"] = (stats["std"] / stats["mean"]).abs()
            st.dataframe(
                stats[["mean", "std", "min", "50%", "max", "CV"]]
                .rename(columns={"mean": "Mean", "std": "Std", "50%": "Median"}),
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
            file_name=f"wb_{int(y_from)}_{int(y_to)}_{'ALL' if 'ALL' in country_list else '-'.join(country_list)}.csv",
            mime="text/csv"
        )

# == TAB 5: AI PHÂN TÍCH ==
with tabs[4]:
    st.header("AI phân tích (tuỳ chọn)")
    df = _get_df()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        if genai is None or not st.secrets.get("GEMINI_API_KEY"):
            st.info("Chưa cấu hình GEMINI_API_KEY trong .streamlit/secrets.toml")
        else:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                sample = df.head(200).to_dict(orient="records")
                prompt = (
                    "Bạn là chuyên gia phân tích kinh tế. Hãy tóm tắt xu hướng chính, điểm bất thường, "
                    "và đưa ra 3 khuyến nghị cho nhà hoạch định chính sách dựa trên dữ liệu sau:\n"
                    f"{sample}"
                )
                with st.spinner("AI đang phân tích..."):
                    resp = model.generate_content(prompt)
                st.markdown(resp.text or "_Không có phản hồi_")
            except Exception as e:
                st.warning(f"AI lỗi: {e}")
