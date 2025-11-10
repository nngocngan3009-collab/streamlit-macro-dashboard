# --- SỬA LỖI SSL (CERTIFICATE) ---
# Đặt 3 dòng này lên ĐẦU TIÊN của file (giống bản cũ) để tránh một số môi trường lỗi SSL
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
# -----------------------------------

from pathlib import Path
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import google.generativeai as genai

# =========================
# 1) CẤU HÌNH ỨNG DỤNG
# =========================
st.set_page_config(
    page_title="Phân tích Kinh tế Vĩ mô (World Bank)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CHƯƠNG TRÌNH THU THẬP & PHÂN TÍCH DỮ LIỆU VĨ MÔ • World Bank API")
st.markdown("---")

WB_BASE = "https://api.worldbank.org/v2"
HEADERS = {"User-Agent": "WB-Streamlit/1.0 (contact: you@example.com)"}

# =========================
# 2) HÀM GỌI API DÙNG CHUNG
# =========================
def _http_get_json(url, params, retries=4, backoff=1.5, timeout=60):
    """
    GET JSON với retry/backoff cho các lỗi 429/5xx.
    Tự ưu tiên header Retry-After khi có.
    """
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            ra = e.response.headers.get("Retry-After") if e.response else None
            sleep_s = max(backoff, int(ra)) if (ra and str(ra).isdigit()) else backoff * (2 ** attempt)
        except requests.RequestException:
            sleep_s = backoff * (2 ** attempt)
        time.sleep(min(sleep_s, 12))
        attempt += 1
    raise RuntimeError("Failed after retries")

# =========================
# 3) HÀM LẤY DANH MỤC
# =========================
@st.cache_data(show_spinner=False)
def list_countries() -> pd.DataFrame:
    """Trả về DataFrame [code, name], lọc bỏ aggregates (region.id == 'NA')."""
    out, page = [], 1
    while True:
        js = _http_get_json(f"{WB_BASE}/country", {"format":"json","per_page":400,"page":page})
        if not isinstance(js, list) or len(js) < 2: break
        meta, data = js
        per_page, total = int(meta.get("per_page", 0)), int(meta.get("total", 0))
        for c in data:
            if (c.get("region") or {}).get("id") != "NA":
                out.append({"code": c["id"], "name": c["name"]})
        if page * per_page >= total: break
        page += 1
    return pd.DataFrame(out).sort_values("name").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def search_indicators(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    """
    Tìm indicator theo từ khóa trong name/ID.
    Trả về DF [id, name, unit, source].
    """
    results, page = [], 1
    key = (keyword or "").strip().lower()
    while page <= max_pages:
        js = _http_get_json(f"{WB_BASE}/indicator", {"format":"json","per_page":5000,"page":page})
        if not isinstance(js, list) or len(js) < 2: break
        meta, data = js
        per_page, total = int(meta.get("per_page", 0)), int(meta.get("total", 0))
        for it in data:
            _id, _name = it.get("id",""), it.get("name","")
            if key and (key not in _name.lower() and key not in _id.lower()):
                continue
            results.append({
                "id": _id,
                "name": _name,
                "unit": it.get("unit",""),
                "source": (it.get("source", {}) or {}).get("value","")
            })
        if page * per_page >= total: break
        page += 1
    df = pd.DataFrame(results).drop_duplicates(subset=["id"]).sort_values("name").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def fetch_series(country_code: str, indicator_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Lấy chuỗi thời gian cho 1 quốc gia + 1 indicator → DataFrame long: [Year, Country, IndicatorID, Value]
    """
    js = _http_get_json(
        f"{WB_BASE}/country/{country_code}/indicator/{indicator_id}",
        {"format":"json", "per_page":20000, "date": f"{int(year_from)}:{int(year_to)}"}
    )
    if not isinstance(js, list) or len(js) < 2:
        return pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    _, data = js
    rows = []
    for d in data:
        rows.append({
            "Year": int(d["date"]) if str(d.get("date","")).isdigit() else None,
            "Country": (d.get("country") or {}).get("value", country_code),
            "IndicatorID": (d.get("indicator") or {}).get("id", indicator_id),
            "Value": d.get("value", None)
        })
    df = pd.DataFrame(rows).dropna(subset=["Year"]).sort_values("Year")
    return df

def pivot_wide(df_long: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    """
    Đổi từ long → wide, map IndicatorID → tên cho dễ đọc.
    """
    if df_long.empty: return pd.DataFrame()
    df = df_long.copy()
    df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(index=["Year","Country"], columns="IndicatorName", values="Value", aggfunc="first")
    return wide.reset_index().sort_values(["Country","Year"]).set_index("Year")

# =========================
# 4) SIDEBAR: CHỌN THAM SỐ
# =========================
st.sidebar.header("Thiết lập")

# Quốc gia
countries_df = list_countries()
selected_country_name = st.sidebar.selectbox(
    "Chọn quốc gia",
    countries_df["name"].tolist(),
    index=(countries_df["name"].tolist().index("Viet Nam") if "Viet Nam" in countries_df["name"].tolist() else 0)
)
country_code = countries_df.loc[countries_df["name"] == selected_country_name, "code"].iloc[0]

# Khoảng năm
st.sidebar.subheader("Khoảng năm")
selected_start_year = st.sidebar.number_input("Từ năm", 1960, 2100, 2004)
selected_end_year = st.sidebar.number_input("Đến năm", 1960, 2100, 2024)
if selected_end_year < selected_start_year:
    st.sidebar.error("Khoảng năm không hợp lệ (Đến năm < Từ năm).")

# Tìm indicator
st.sidebar.subheader("Chỉ số (indicator)")
kw = st.sidebar.text_input("Tìm theo từ khóa (ví dụ: GDP, inflation, unemployment...)", value="GDP")
with st.sidebar:
    if st.button("Tìm chỉ số"):
        st.session_state["indicator_search"] = kw

# Lấy kết quả tìm kiếm (cache theo từ khóa)
search_key = st.session_state.get("indicator_search", kw)
ind_df = search_indicators(search_key, max_pages=2)
if ind_df.empty:
    st.sidebar.info("Không tìm thấy indicator theo từ khóa.")
ind_options = [f'{r["id"]} — {r["name"]}' for _, r in ind_df.iterrows()]

# Multiselect indicator (chọn 1 hoặc nhiều)
selected_indicators_pretty = st.sidebar.multiselect(
    "Chọn indicator (có thể chọn nhiều):",
    options=ind_options,
    default=[x for x in ind_options if x.startswith("NY.GDP.MKTP.CD")][:1]  # mặc định GDP USD nếu có
)
selected_indicator_ids = [opt.split(" — ", 1)[0] for opt in selected_indicators_pretty]

# Xử lý dữ liệu
st.sidebar.subheader("Xử lý N/A")
handling_method = st.sidebar.selectbox(
    "Áp dụng cho toàn bộ chỉ số:",
    ["Giữ nguyên (N/A)", "Điền giá trị gần nhất (Forward Fill)", "Điền trung bình theo cột (Mean)"]
)

# =========================
# 5) TẢI DỮ LIỆU THEO LỰA CHỌN
# =========================
@st.cache_data(show_spinner=True)
def get_data(country_code: str, indicator_ids: list, y0: int, y1: int) -> pd.DataFrame:
    """
    Tải tất cả indicator đã chọn cho 1 quốc gia → DF wide (index=Year).
    """
    if not indicator_ids:
        return pd.DataFrame()

    # Lấy chuỗi từng indicator rồi gộp
    all_long = []
    id_to_name = {}
    # map id → name (từ kết quả tìm kiếm)
    if not ind_df.empty:
        id_to_name.update({row["id"]: row["name"] for _, row in ind_df.iterrows()})
    # fallback nếu user nhập ID không có trong trang kết quả
    for ind in indicator_ids:
        long_df = fetch_series(country_code, ind, y0, y1)
        all_long.append(long_df)
        id_to_name.setdefault(ind, ind)
        time.sleep(0.2)  # tránh 429 khi chọn nhiều

    full_long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    wide = pivot_wide(full_long, id_to_name)  # index=Year, có cột Country + các chỉ số
    # Lọc đúng quốc gia (đề phòng WB trả về thêm)
    if not wide.empty and "Country" in wide.columns:
        wide = wide[wide["Country"] == selected_country_name]
        wide = wide.drop(columns=["Country"])
    # Áp dụng xử lý N/A
    if handling_method == "Điền giá trị gần nhất (Forward Fill)":
        wide = wide.ffill()
    elif handling_method == "Điền trung bình theo cột (Mean)":
        wide = wide.apply(lambda x: x.fillna(x.mean()), axis=0)
    return wide

if selected_indicator_ids and selected_end_year >= selected_start_year:
   df_wide = get_data(
    country_code,
    selected_indicator_ids,
    selected_start_year,
    selected_end_year,
    selected_country_name,   # <— thêm
    handling_method          # <— thêm
)
else:
    df_wide = pd.DataFrame()

# =========================
# 6) TABS: DỮ LIỆU • BIỂU ĐỒ • STATS • DOWNLOAD • AI
# =========================
tab_data, tab_charts, tab_stats, tab_download, tab_ai = st.tabs([
    "📊 Dữ liệu",
    "📈 Biểu đồ",
    "🧮 Thống kê mô tả",
    "📥 Tải dữ liệu",
    "🤖 AI phân tích và tư vấn"
])

# == TAB 1: DỮ LIỆU ==
with tab_data:
    st.header(f"Bảng dữ liệu — {selected_country_name} ({selected_start_year}-{selected_end_year})")
    st.info(f"Đang áp dụng xử lý N/A: **{handling_method}**.")
    if df_wide.empty:
        st.warning("Chưa có dữ liệu. Hãy chọn ít nhất 1 indicator.")
    else:
        st.dataframe(df_wide.style.format("{:.2f}", na_rep="N/A"))

# == TAB 2: BIỂU ĐỒ ==
with tab_charts:
    st.header("Trực quan hoá dữ liệu")
    if df_wide.empty:
        st.warning("Không có dữ liệu để vẽ.")
    else:
        all_cols = df_wide.columns.tolist()
        # LINE
        st.subheader("Biểu đồ đường (Line)")
        line_cols = st.multiselect("Chọn chỉ số cho Line:", options=all_cols, default=all_cols[:min(4, len(all_cols))])
        if line_cols:
            fig_line = px.line(df_wide.reset_index(), x="Year", y=line_cols, title=f"Xu hướng tại {selected_country_name}")
            fig_line.update_layout(xaxis_title="Năm", yaxis_title="Giá trị")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Chọn ít nhất 1 chỉ số để vẽ Line.")
        st.markdown("---")
        # BAR
        st.subheader("Biểu đồ cột (Bar)")
        bar_cols = st.multiselect("Chọn chỉ số cho Bar:", options=all_cols, default=line_cols)
        if bar_cols:
            fig_bar = px.bar(df_wide.reset_index(), x="Year", y=bar_cols, title=f"Biểu đồ cột tại {selected_country_name}", barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Chọn ít nhất 1 chỉ số để vẽ Bar.")
        st.markdown("---")
        # HEATMAP
        st.subheader("Heatmap tương quan")
        heat_cols = st.multiselect("Chọn chỉ số cho Heatmap:", options=all_cols, default=all_cols[:min(4, len(all_cols))])
        if len(heat_cols) > 1:
            corr = df_wide[heat_cols].fillna(0).corr()
            fig_hm = ff.create_annotated_heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale='Viridis',
                annotation_text=corr.round(2).values
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Cần ít nhất 2 chỉ số để vẽ heatmap.")

# == TAB 3: THỐNG KÊ MÔ TẢ ==
with tab_stats:
    st.header("Bảng thống kê mô tả")
    if df_wide.empty:
        st.warning("Không có dữ liệu để thống kê.")
    else:
        stats = df_wide.describe().transpose()
        if not stats.empty:
            if 'count' in stats.columns:
                stats = stats.drop(columns=['count'])
            stats = stats.rename(columns={
                'mean': 'Giá trị TB (Mean)', 'std': 'Độ lệch chuẩn (Std)',
                'min': 'Nhỏ nhất (Min)', 'max': 'Lớn nhất (Max)',
                '50%': 'Trung vị (Median)'
            })
            stats['Hệ số biến thiên (CV)'] = (stats['Độ lệch chuẩn (Std)'] / stats['Giá trị TB (Mean)']).abs()
            st.dataframe(
                stats[['Giá trị TB (Mean)','Độ lệch chuẩn (Std)','Nhỏ nhất (Min)','Lớn nhất (Max)','Trung vị (Median)','25%','75%','Hệ số biến thiên (CV)']].style.format("{:.3f}")
            )
        else:
            st.warning("Không thể tính thống kê (có thể toàn N/A).")

# == TAB 4: TẢI DỮ LIỆU ==
with tab_download:
    st.header(f"Tải dữ liệu — {selected_country_name}")
    if df_wide.empty:
        st.info("Không có dữ liệu để tải.")
    else:
        @st.cache_data
        def to_csv_bytes(df: pd.DataFrame):
            return df.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")
        csv_bytes = to_csv_bytes(df_wide)
        fn = f"worldbank_{country_code.lower()}_{selected_start_year}_{selected_end_year}.csv"
        st.download_button("📥 Tải CSV", data=csv_bytes, file_name=fn, mime="text/csv")
        st.info("CSV UTF-8-SIG để mở bằng Excel không lỗi font.")

# == TAB 5: AI (Gemini) ==
with tab_ai:
    st.header("AI phân tích và tư vấn")
    target_audience = "Ngân hàng Agribank"
    st.subheader(f"Đối tượng tư vấn: {target_audience}")

    def generate_ai_analysis(data_df: pd.DataFrame, country: str, audience: str):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            # Dùng model ổn định, sẵn có rộng rãi
            model = genai.GenerativeModel("gemini-2.5-pro")
            data_string = data_df.reset_index().to_csv(index=False)

            prompt = f"""
Bạn là chuyên gia kinh tế vĩ mô. Dữ liệu sau là của **{country}** giai đoạn {selected_start_year}-{selected_end_year} (từ World Bank):

{data_string}

Yêu cầu: Viết báo cáo gồm 5 phần:
1) Bối cảnh & Dữ liệu chính (nêu các chỉ số nổi bật).
2) Xu hướng nổi bật & năm biến động mạnh (kèm diễn giải ngắn).
3) Tương quan đáng chú ý giữa các chỉ số (nếu có).
4) 3–4 kiến nghị cho đối tượng: **{audience}** (ưu tiên bối cảnh Việt Nam nếu phù hợp).
5) Hành động thực thi: 1–2 hành động kèm KPI hoặc “điều kiện kích hoạt”.

Trình bày súc tích, có tiêu đề phụ, bullet khi cần.
"""
            with st.spinner(f"AI đang phân tích {country} cho {audience}..."):
                resp = model.generate_content(prompt)
                return resp.text
        except Exception as e:
            if "API_key" in str(e):
                st.error("Không tìm thấy GEMINI_API_KEY. Hãy thêm vào .streamlit/secrets.toml")
            elif "invalid" in str(e).lower():
                st.error("GEMINI_API_KEY không hợp lệ.")
            else:
                st.error(f"Lỗi gọi Gemini: {e}")
            return None

    if st.button(f"🚀 Sinh AI phân tích và tư vấn cho {target_audience}"):
        if df_wide.empty:
            st.error("Không có dữ liệu để phân tích. Hãy chọn indicator trước.")
        else:
            report = generate_ai_analysis(df_wide, selected_country_name, target_audience)
            if report:
                st.markdown(report)
