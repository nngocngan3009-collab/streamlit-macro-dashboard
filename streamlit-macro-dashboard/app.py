# =========================
# python7_fixed.py — Data360 search (WB_WDI) + World Bank fetch
# - Search indicators qua Data360 (lọc WB_WDI, chuẩn ID hiển thị)
# - Bảng dữ liệu đầu ra DẠNG RỘNG:
#   Year | Country | <Indicator Name 1> | <Indicator Name 2> | ...
# - Bảo đảm: chọn bao nhiêu indicator → có bấy nhiêu cột (kể cả rỗng nếu không có dữ liệu)
# - Tính năng: Biểu đồ, Heatmap, Thống kê, CSV, AI (Gemini) như bản chuẩn
# =========================

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import time
from typing import Dict, Any, Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff

# (tuỳ chọn) AI insight
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- Config ----------------
WB_BASE = "https://api.worldbank.org/v2"
DATA360_BASE_URL = os.environ.get("DATA360_BASE_URL", "https://api.data360.org")
D360_SEARCH_ENDPOINT = "/data360/searchv2"

HEADERS = {
    "User-Agent": "Streamlit-WB-Client/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}
REQ_TIMEOUT = 60
RETRIES = 4
BACKOFF = 1.6

DEFAULT_DATE_RANGE = (2004, 2024)

# ---------------- Retry helpers ----------------
def _sleep(attempt: int) -> float:
    return min(BACKOFF ** attempt, 12.0)

def http_get_json(url: str, params: Dict[str, Any]) -> Any:
    last_err = None
    for i in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(i))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")

def data360_request_json(payload: Dict[str, Any]) -> Any:
    url = f"{DATA360_BASE_URL}{D360_SEARCH_ENDPOINT}"
    last_err = None
    for i in range(RETRIES + 1):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(i))
    raise RuntimeError(f"POST {url} failed after retries: {last_err}")

# ---------------- Utilities ----------------
def _to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _format_dot(code_underscore: str) -> str:
    """
    SP_POP_TOTL -> SP.POP.TOTL
    """
    return (code_underscore or "").strip("_").replace("_", ".")

def _cut_wb_id(full_id: str) -> str:
    """
    WB_WDI_SP_POP_TOTL -> SP_POP_TOTL
    (Nếu không đúng format thì trả nguyên)
    """
    s = full_id or ""
    return s[len("WB_WDI_"):] if s.startswith("WB_WDI_") else s

# ---------------- Country list ----------------
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
            # Bỏ nhóm không áp dụng ("Aggregates": region id = "NA")
            if (c.get("region") or {}).get("id") != "NA":
                out.append({"code": c["id"], "name": c["name"]})
        if page * per_page >= total:
            break
        page += 1
    return pd.DataFrame(out).sort_values("name").reset_index(drop=True)

# ---------------- Indicator search (Data360 first, WB fallback) ----------------
@st.cache_data(show_spinner=False, ttl=6*3600)
def wb_indicator_catalog(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    """
    Fallback: World Bank /indicator
    """
    results, page = [], 1
    key = (keyword or "").strip().lower()
    while page <= max_pages:
        js = http_get_json(f"{WB_BASE}/indicator", {"format":"json","per_page":5000,"page":page})
        if not isinstance(js, list) or len(js) < 2:
            break
        meta, data = js
        per_page, total = _to_int(meta.get("per_page",0)), _to_int(meta.get("total",0))
        for it in data:
            _id, _name = it.get("id",""), it.get("name","")
            if key and (key not in _name.lower() and key not in _id.lower()):
                continue
            results.append({"id": _id, "name": _name})
        if page * per_page >= total:
            break
        page += 1
    df = pd.DataFrame(results).drop_duplicates(subset=["id"]).sort_values("name").reset_index(drop=True)
    # Chuẩn cột để đồng nhất với Data360 schema hiển thị
    if df.empty:
        return pd.DataFrame(columns=["display_code","name","full_id","wb_id"])
    df["display_code"] = df["id"]         # đã là dot format
    df["full_id"] = df["id"]
    df["wb_id"] = "WB_WDI"
    return df[["display_code","name","wb_id","full_id"]]

@st.cache_data(show_spinner=False, ttl=6*3600)
def data360_search_indicators(keyword: str, top: int = 40) -> pd.DataFrame:
    """
    Search indicators via Data360 searchv2, filter only WB_WDI.
    Chuẩn hoá hiển thị 'display_code' dạng SP.POP.TOTL / NY.GDP.MKTP.CD.
    Loại bỏ các kết quả kiểu '6.0.GDP_usd' bằng cách:
      - chỉ giữ record có series_description/database_id == 'WB_WDI'
      - lấy short_id từ full_id 'WB_WDI_SP_POP_TOTL' rồi format dot.
    """
    payload = {
        "count": False,
        "search": (keyword or "").strip(),
        "select": "series_description/idno, series_description/name, series_description/database_id",
        "top": max(5, int(top)),
        "filter": "type eq 'indicator' and series_description/database_id eq 'WB_WDI'",
    }
    try:
        js = data360_request_json(payload)
        values = js.get("value", []) if isinstance(js, dict) else []
    except Exception:
        # Fallback WB catalog
        return wb_indicator_catalog(keyword, max_pages=2)

    rows = []
    for item in values:
        sd = item.get("series_description") or {}
        full_id = sd.get("idno") or item.get("series_description/idno", "")
        dbid = sd.get("database_id") or item.get("series_description/database_id", "")
        if dbid != "WB_WDI" or not full_id:
            continue

        short_underscore = _cut_wb_id(full_id)   # SP_POP_TOTL
        display_code = _format_dot(short_underscore)  # SP.POP.TOTL
        name = sd.get("name") or item.get("series_description/name", "")
        if display_code and name:
            rows.append({
                "display_code": display_code,
                "name": name,
                "wb_id": "WB_WDI",
                "full_id": full_id,  # giữ lại để mapping ngược nếu cần
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["display_code"]).sort_values("name").reset_index(drop=True)
    if df.empty:
        # Fallback WB catalog
        return wb_indicator_catalog(keyword, max_pages=2)
    return df

# ---------------- Fetch data (World Bank v2) ----------------
@st.cache_data(show_spinner=False, ttl=60*30)
def wb_fetch_series(country_code: str, wb_dot_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Trả về DF cột: Year, Country, IndicatorID, Value
    wb_dot_id ví dụ: NY.GDP.MKTP.CD
    """
    js = http_get_json(
        f"{WB_BASE}/country/{country_code}/indicator/{wb_dot_id}",
        {"format": "json", "per_page": 20000, "date": f"{year_from}:{year_to}"}
    )

    # Defensive
    if not isinstance(js, list) or len(js) < 2:
        return pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    if isinstance(js[0], dict) and js[0].get("message"):
        return pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    _, data = js
    if not isinstance(data, list):
        return pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])

    rows = []
    for d in data:
        year = d.get("date")
        if not str(year).isdigit():
            continue
        rows.append({
            "Year": int(year),
            "Country": (d.get("country") or {}).get("value", country_code),
            "IndicatorID": (d.get("indicator") or {}).get("id", wb_dot_id),
            "Value": d.get("value", None),
        })
    out = pd.DataFrame(rows).dropna(subset=["Year"])
    return out.sort_values("Year") if not out.empty else pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])

def pivot_wide_with_missing(df_long: pd.DataFrame, id_to_name: Dict[str,str], expected_names: List[str]) -> pd.DataFrame:
    """
    Pivot long -> wide (Year, Country, columns by IndicatorName).
    Đảm bảo mọi 'expected_names' đều là cột trong wide, kể cả nếu thiếu dữ liệu (điền NaN).
    """
    if df_long is None or df_long.empty:
        # Trả khung trống với đầy đủ cột
        cols = ["Year","Country"] + list(expected_names)
        return pd.DataFrame(columns=cols)

    df = df_long.copy()
    df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(index=["Year","Country"], columns="IndicatorName", values="Value", aggfunc="first")
    wide = wide.reset_index().sort_values(["Country","Year"])

    # Bổ sung cột còn thiếu
    for col in expected_names:
        if col not in wide.columns:
            wide[col] = np.nan

    # Sắp xếp cột: Year, Country, rồi theo danh sách đầu vào
    ordered = ["Year","Country"] + [c for c in expected_names]
    return wide[ordered]

# ---------------- NA handling ----------------
def handle_na(df: pd.DataFrame, na_method: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if na_method == "Giữ nguyên (N/A)":
        return df
    if na_method == "Điền giá trị gần nhất (Forward Fill)":
        return df.sort_values(["Country","Year"]).groupby("Country").ffill()
    if na_method == "Điền trung bình theo cột (Mean)":
        num = df.select_dtypes(include=[np.number])
        df[num.columns] = num.apply(lambda x: x.fillna(x.mean()), axis=0)
        return df
    return df

# ---------------- UI ----------------
st.set_page_config(page_title="Data360 → World Bank (WB_WDI) — Fixed", layout="wide", initial_sidebar_state="expanded")
st.title("🔎 Data360 → World Bank (WB_WDI) — Fixed")
st.caption("Search indicator qua Data360 (lọc WB_WDI, ID chuẩn). Bảng dữ liệu đầu ra dạng rộng: **Year | Country | ...**")

# Sidebar: Country + Year range
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
selected_country_name = country_display.split("(")[0].strip()

min_year, max_year = DEFAULT_DATE_RANGE
c1, c2 = st.sidebar.columns(2)
y_from = c1.number_input("Từ năm", min_value=1960, max_value=2100, value=min_year, step=1)
y_to   = c2.number_input("Đến năm", min_value=1960, max_value=2100, value=max_year, step=1)

# Sidebar: Search indicators via Data360
st.sidebar.subheader("Tìm & chọn chỉ số (Data360 → WB_WDI)")
kw = st.sidebar.text_input("Từ khoá (vd: GDP, CPI, inflation...)", value="GDP")

# NA method
st.sidebar.subheader("Xử lý dữ liệu (N/A)")
na_method = st.sidebar.selectbox(
    "Phương án xử lý N/A",
    ["Giữ nguyên (N/A)", "Điền giá trị gần nhất (Forward Fill)", "Điền trung bình theo cột (Mean)"],
    index=0
)

if "ind_df_cache_d360" not in st.session_state:
    st.session_state["ind_df_cache_d360"] = pd.DataFrame()

if st.sidebar.button("🔍 Tìm chỉ số"):
    with st.spinner("Đang tìm indicators từ Data360 (lọc WB_WDI)…"):
        st.session_state["ind_df_cache_d360"] = data360_search_indicators(kw, top=40)

ind_df = st.session_state["ind_df_cache_d360"]
with st.sidebar.expander("Kết quả tìm thấy", expanded=False):
    if ind_df.empty:
        st.info("Nhấn **Tìm chỉ số** để tra cứu.")
    else:
        st.dataframe(
            ind_df.rename(columns={
                "display_code": "Indicator",
                "name": "Tên chỉ số",
                "wb_id": "WB_ID",
                "full_id": "Full ID"
            })[["Indicator","Tên chỉ số","WB_ID","Full ID"]],
            use_container_width=True, height=260
        )

# Chọn theo tên (để hiển thị cột đúng chuẩn)
indicator_names = ind_df["name"].tolist() if not ind_df.empty else []
selected_indicator_names = st.sidebar.multiselect(
    "Chọn **tên** chỉ số để lấy dữ liệu",
    options=indicator_names,
    default=indicator_names[:2] if indicator_names else []
)

# Mapping: name -> dot id (SP.POP.TOTL), id_to_name (dot id -> *Tên chỉ số*)
name_to_dot = {row["name"]: row["display_code"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}
dot_to_name = {v: k for k, v in name_to_dot.items()}

# ---------------- Tabs ----------------
tabs = st.tabs(["📊 Dữ liệu","📈 Biểu đồ","🧮 Thống kê","📥 Tải CSV","🤖 AI"])

# == TAB 1: DỮ LIỆU ==
with tabs[0]:
    st.subheader("Dữ liệu (dạng rộng)")
    if st.button("📥 Lấy dữ liệu"):
        if not selected_indicator_names:
            st.warning("Chọn ít nhất một *tên* chỉ số.")
            st.stop()

        chosen_dot_ids = [name_to_dot[n] for n in selected_indicator_names if n in name_to_dot]

        all_long = []
        with st.spinner(f"Tải {len(chosen_dot_ids)} chỉ số cho {country_code}…"):
            for iid in chosen_dot_ids:
                df_fetch = wb_fetch_series(country_code, iid, int(y_from), int(y_to))
                if df_fetch is not None and not df_fetch.empty:
                    all_long.append(df_fetch)
                else:
                    # Nếu không có dữ liệu, vẫn tạo khung rỗng để đảm bảo có cột sau pivot
                    all_long.append(pd.DataFrame(columns=["Year","Country","IndicatorID","Value"]))

                time.sleep(0.25)  # nho nhỏ tránh đụng rate limit

        # Gộp long
        if not all_long:
            st.error("Không có dữ liệu phù hợp cho phạm vi năm/chỉ số đã chọn.")
            st.stop()

        df_long = pd.concat(all_long, ignore_index=True) if len(all_long) > 1 else (all_long[0] if all_long else pd.DataFrame())
        # Đảm bảo IndicatorID là dot id (đúng chuẩn)
        if not df_long.empty:
            # Một số API trả indicator.id sẵn dot id, nhưng cứ chuẩn hoá tên map
            pass

        # Pivot dạng rộng + BỔ SUNG cột thiếu
        expected_names = [n for n in selected_indicator_names]  # cột cần có theo tên (label)
        id_to_name = {dot: dot_to_name.get(dot, dot) for dot in chosen_dot_ids}  # map dot → name
        df_wide = pivot_wide_with_missing(df_long, id_to_name, expected_names)

        # Xử lý N/A theo tuỳ chọn
        df_wide = handle_na(df_wide, na_method)

        st.session_state["wb_df_wide"] = df_wide
        st.success("✅ Đã tải và chuẩn hoá dữ liệu.")
        st.dataframe(df_wide.set_index("Year"), use_container_width=True)

def _get_df_wide():
    return st.session_state.get("wb_df_wide", pd.DataFrame())

# == TAB 2: BIỂU ĐỒ ==
with tabs[1]:
    st.subheader("Biểu đồ")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải.")
    else:
        cols = [c for c in df.columns if c not in ("Year","Country")]
        choose = st.multiselect("Chọn cột vẽ", options=cols, default=cols[:min(4, len(cols))])
        if choose:
            st.plotly_chart(px.line(df, x="Year", y=choose, color="Country", markers=True, title="Xu hướng"), use_container_width=True)

            if len(choose) > 1:
                df_sel = df[choose].apply(pd.to_numeric, errors="coerce")
                df_sel = df_sel.dropna(axis=1, how="all")
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
                else:
                    st.info("Các cột được chọn không đủ dữ liệu số để tính tương quan.")

# == TAB 3: THỐNG KÊ ==
with tabs[2]:
    st.subheader("Thống kê mô tả")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        cols = [c for c in df.columns if c not in ("Year","Country")]
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

# == TAB 4: CSV ==
with tabs[3]:
    st.subheader("Tải CSV")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.download_button(
            "📥 Tải CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"wb_{country_code}_{y_from}_{y_to}.csv",
            mime="text/csv"
        )

# == TAB 5: AI ==
with tabs[4]:
    st.subheader("AI insight (tuỳ chọn)")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        if genai is None or not os.environ.get("GOOGLE_API_KEY", ""):
            st.info("Chưa cấu hình GOOGLE_API_KEY nên bỏ qua AI insight.")
        else:
            try:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                model = genai.GenerativeModel("gemini-2.5-flash")
                data_string = df.to_csv(index=False)
                prompt = (
                    "Bạn là chuyên gia dữ liệu kinh tế. Hãy tóm tắt xu hướng chính, điểm bất thường, "
                    f"và gợi ý 2–3 insight hành động cho quốc gia {selected_country_name} "
                    f"trong giai đoạn {y_from}-{y_to}. Dữ liệu CSV:\n\n{data_string}\n\n"
                    "Trả lời ngắn gọn, dạng bullet."
                )
                with st.spinner("AI đang phân tích…"):
                    resp = model.generate_content(prompt)
                st.markdown(resp.text or "_Không có phản hồi_")
            except Exception as e:
                st.warning(f"AI lỗi: {e}")
