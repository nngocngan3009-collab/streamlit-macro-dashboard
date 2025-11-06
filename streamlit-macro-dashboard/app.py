# --- SỬA LỖI SSL (CERTIFICATE) ---
# Đặt 3 dòng này lên ĐẦU TIÊN của file
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
# -----------------------------------
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import google.generativeai as genai

# --- 1. CẤU HÌNH TRANG VÀ TIÊU ĐỀ ---
st.set_page_config(
    page_title="Phân tích Kinh tế Vĩ mô",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề chính
st.title("CHƯƠNG TRÌNH THU THẬP VÀ TỔNG HỢP THÔNG TIN KINH TẾ VĨ MÔ")
st.markdown("---")

# --- 2. HÀM TẢI DỮ LIỆU TỪ FILE CSV ---
@st.cache_data
def load_data():
    """Tải dữ liệu từ file macro_data.csv (cùng thư mục với app.py)"""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "macro_data.csv"
    try:
        df = pd.read_csv(csv_path, na_values="N/A")
        return df
    except FileNotFoundError:
        st.error(
            f"Lỗi: Không tìm thấy file '{csv_path.name}'. "
            f"Đường dẫn đang tìm: {csv_path}"
        )
        return pd.DataFrame() 

# Tải toàn bộ dữ liệu
df_all_data = load_data()

# --- 3. KHU VỰC THANH BÊN (SIDEBAR) ĐỂ LỌC ---
st.sidebar.header("Thiết lập")

if df_all_data.empty:
    st.sidebar.warning("Không thể tải dữ liệu.")
else:
    # BỘ LỌC 1: QUỐC GIA
    st.sidebar.subheader("Quốc gia")
    all_countries = df_all_data['Country'].unique()
    selected_country = st.sidebar.selectbox("Chọn quốc gia", all_countries)

    # Lọc data theo quốc gia đã chọn
    df_country = df_all_data[df_all_data['Country'] == selected_country].copy()
    df_country.set_index('Year', inplace=True) 
    
    # BỘ LỌC 2: KHOẢNG NĂM
    min_year = int(df_country.index.min())
    max_year = int(df_country.index.max())

    st.sidebar.subheader("Khoảng năm")
    selected_start_year = st.sidebar.number_input("Từ năm", min_year, max_year, min_year)
    selected_end_year = st.sidebar.number_input("Đến năm", min_year, max_year, max_year)

    # Lọc dữ liệu theo năm
    df_filtered = df_country.loc[selected_start_year:selected_end_year]

    # BỘ LỌC 3: CHỈ SỐ
    st.sidebar.subheader("Chỉ số")
    all_indicators = df_country.columns.drop('Country', errors='ignore')
    
    # *** THAY ĐỔI: Đổi tên biến để rõ nghĩa ***
    selected_indicators_sidebar = st.sidebar.multiselect(
        "Chọn chỉ số (mặc định)",
        all_indicators,
        default=all_indicators[:4].tolist() 
    )

    # *** THAY ĐỔI: Chuyển logic xử lý N/A từ Tab Dữ liệu ra Sidebar ***
    st.sidebar.subheader("Xử lý dữ liệu")
    handling_method = st.sidebar.selectbox(
        "Phương án xử lý N/A (Áp dụng cho tất cả)",
        ["Giữ nguyên (N/A)", "Điền giá trị gần nhất (Forward Fill)", "Điền trung bình theo cột (Mean)"]
    )

    # --- TẠO 2 DATAFRAME ĐÃ XỬ LÝ (TRƯỚC KHI VÀO TAB) ---
    
    # 1. DataFrame CHỈ chứa các cột được CHỌN Ở SIDEBAR (dùng cho Data, Stats, AI, Download)
    df_selected_cols = df_filtered[selected_indicators_sidebar]
    df_processed_sidebar = df_selected_cols.copy()
    if handling_method == "Điền giá trị gần nhất (Forward Fill)":
        df_processed_sidebar = df_processed_sidebar.ffill()
    elif handling_method == "Điền trung bình theo cột (Mean)":
        df_processed_sidebar = df_processed_sidebar.apply(lambda x: x.fillna(x.mean()), axis=0)

    # 2. DataFrame chứa TOÀN BỘ các cột (dùng cho tab Biểu đồ)
    df_all_cols = df_filtered[all_indicators]
    df_processed_full = df_all_cols.copy()
    if handling_method == "Điền giá trị gần nhất (Forward Fill)":
        df_processed_full = df_processed_full.ffill()
    elif handling_method == "Điền trung bình theo cột (Mean)":
        df_processed_full = df_processed_full.apply(lambda x: x.fillna(x.mean()), axis=0)
    # -----------------------------------------------------------------

    # --- 4. KHU VỰC NỘI DUNG CHÍNH (VỚI CÁC TAB) ---

    tab_data, tab_charts, tab_stats, tab_download, tab_ai = st.tabs([
        "📊 Dữ liệu", 
        "📈 Biểu đồ", 
        "🧮 Thống kê mô tả",
        "📥 Tải dữ liệu",
        "🤖 AI phân tích và tư vấn"
    ])

    # == TAB 1: DỮ LIỆU VÀ XỬ LÝ ==
    with tab_data:
        st.header(f"Bảng dữ liệu đã xử lý - {selected_country} ({selected_start_year}-{selected_end_year})")
        
        # *** THAY ĐỔI: Xóa bộ lọc N/A ở đây, thêm thông báo ***
        st.info(f"Đang áp dụng phương án xử lý N/A: **{handling_method}** (được chọn ở thanh bên).")
        st.write("Bảng này chỉ hiển thị các chỉ số bạn đã chọn ở thanh bên.")
        
        # *** THAY ĐỔI: Sử dụng df_processed_sidebar ***
        st.dataframe(df_processed_sidebar.style.format("{:.2f}", na_rep="N/A"))

    # == TAB 2: BIỂU ĐỒ TRỰC QUAN HÓA ==
    with tab_charts:
        st.header("Trực quan hoá dữ liệu")
        
        # *** THAY ĐỔI LỚN: Cấu trúc lại toàn bộ tab này ***
        if not all_indicators.any():
            st.warning("Không có dữ liệu chỉ số nào cho quốc gia này.")
        else:
            st.info("Tại đây, mỗi biểu đồ có thể chọn bộ chỉ số riêng. Mặc định là các chỉ số bạn đã chọn ở thanh bên.")

            # --- KHU VỰC BIỂU ĐỒ LINE ---
            st.subheader("Biểu đồ xu hướng theo thời gian (Line chart)")
            line_indicators = st.multiselect(
                "Chọn chỉ số cho biểu đồ Line:",
                options=all_indicators,           # Req 3: Full các chỉ tiêu
                default=selected_indicators_sidebar # Req 1: Mặc định là chỉ tiêu ở sidebar
            )
            
            if line_indicators:
                fig_line = px.line(df_processed_full, # Dùng df full
                                   x=df_processed_full.index, 
                                   y=line_indicators, # Dùng list chỉ số riêng
                                   title=f"Xu hướng các chỉ số tại {selected_country}")
                fig_line.update_layout(xaxis_title="Năm", yaxis_title="Giá trị")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("Vui lòng chọn ít nhất một chỉ số cho biểu đồ Line.")

            st.markdown("---") # Ngăn cách

            # --- KHU VỰC BIỂU ĐỒ BAR ---
            st.subheader("Biểu đồ cột (Bar chart)")
            bar_indicators = st.multiselect(
                "Chọn chỉ số cho biểu đồ Cột:",
                options=all_indicators,
                default=selected_indicators_sidebar
            )
            
            if bar_indicators:
                fig_bar = px.bar(df_processed_full, # Dùng df full
                                 x=df_processed_full.index, 
                                 y=bar_indicators, # Dùng list chỉ số riêng
                                 title=f"Biểu đồ cột các chỉ số tại {selected_country}", 
                                 barmode="group")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Vui lòng chọn ít nhất một chỉ số cho biểu đồ Cột.")
            
            st.markdown("---") # Ngăn cách

            # --- KHU VỰC BIỂU ĐỒ HEATMAP ---
            st.subheader("Heatmap tương quan giữa các chỉ số")
            heatmap_indicators = st.multiselect(
                "Chọn chỉ số cho Heatmap:",
                options=all_indicators,
                default=selected_indicators_sidebar
            )

            if len(heatmap_indicators) > 1:
                # Dùng df full và list chỉ số riêng
                corr_matrix = df_processed_full[heatmap_indicators].fillna(0).corr() 
                fig_heatmap = ff.create_annotated_heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale='Viridis',
                    annotation_text=corr_matrix.round(2).values
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Cần ít nhất 2 chỉ số để vẽ heatmap tương quan.")

    # == TAB 3: THỐNG KÊ MÔ TẢ ==
    with tab_stats:
        st.header("Bảng thống kê mô tả")
        st.write(f"Thống kê cho các chỉ số của {selected_country} ({selected_start_year}-{selected_end_year}), sau khi {handling_method}.")
        
        # *** THAY ĐỔI: Dùng biến và df mới ***
        if not selected_indicators_sidebar:
            st.warning("Vui lòng chọn ít nhất một chỉ số ở thanh bên.")
        else:
            stats = df_processed_sidebar.describe().transpose() # Dùng df_processed_sidebar
            
            if not stats.empty:
                stats = stats.drop(columns=['count']) 
                stats['Hệ số biến thiên (CV)'] = (stats['std'] / stats['mean']).abs()
                stats['Năm nhỏ nhất (Min)'] = df_processed_sidebar.idxmin() # Dùng df_processed_sidebar
                stats['Năm lớn nhất (Max)'] = df_processed_sidebar.idxmax() # Dùng df_processed_sidebar
                
                stats = stats.rename(columns={
                    'mean': 'Giá trị TB (Mean)', 'std': 'Độ lệch chuẩn (Std)',
                    'min': 'Nhỏ nhất (Min)', 'max': 'Lớn nhất (Max)',
                    '50%': 'Trung vị (Median)'
                })
                
                column_order = [
                    'Giá trị TB (Mean)', 'Độ lệch chuẩn (Std)', 
                    'Nhỏ nhất (Min)', 'Năm nhỏ nhất (Min)',
                    'Lớn nhất (Max)', 'Năm lớn nhất (Max)',
                    'Trung vị (Median)', '25%', '75%', 'Hệ số biến thiên (CV)'
                ]
                
                final_columns = [col for col in column_order if col in stats.columns]
                stats_final = stats[final_columns]
                
                st.dataframe(stats_final.style.format("{:.3f}", 
                               subset=['Giá trị TB (Mean)', 'Độ lệch chuẩn (Std)', 'Nhỏ nhất (Min)', 'Lớn nhất (Max)', 'Trung vị (Median)', '25%', '75%', 'Hệ số biến thiên (CV)'])
                               .format("{:d}", subset=['Năm nhỏ nhất (Min)', 'Năm lớn nhất (Max)'], na_rep="N/A"))
            else:
                st.warning("Không thể tính toán thống kê. Dữ liệu có thể toàn N/A.")

    # == TAB 4: TẢI DỮ LIỆU ==
    with tab_download:
        st.header(f"Tải về dữ liệu cho {selected_country}")
        st.write(f"Dữ liệu này đã được lọc theo năm ({selected_start_year}-{selected_end_year}) và đã được xử lý N/A theo phương án: **{handling_method}**.")
        st.write("File tải về chỉ chứa các chỉ số bạn đã chọn ở thanh bên.")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')

        # *** THAY ĐỔI: Dùng df_processed_sidebar ***
        csv_data = convert_df_to_csv(df_processed_sidebar)
        
        file_name = f"data_{selected_country.lower().replace(' ', '_')}_{selected_start_year}_{selected_end_year}.csv"
        
        st.download_button(
            label="📥 Tải về file CSV",
            data=csv_data,
            file_name=file_name,
            mime='text/csv',
        )
        st.info("File tải về ở định dạng .csv, bạn có thể mở bằng Excel. File đã được mã hóa UTF-8-SIG để đảm bảo không lỗi font tiếng Việt.")
    
    # == TAB 5: AI PHÂN TÍCH VÀ TƯ VẤN ==
    with tab_ai:
        st.header("AI phân tích và tư vấn")
        
        target_audience = "Ngân hàng Agribank"
        st.subheader(f"Đối tượng tư vấn: {target_audience}") # Hiển thị cho người dùng biết
        
        # Hàm gọi AI (không đổi)
        def generate_ai_analysis(data_df, country, audience):
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                genai.configure(api_key=api_key)
                
                model = genai.GenerativeModel('gemini-2.5-pro') # Sửa lại model nếu cần
                data_string = data_df.to_csv()
        
                prompt_template = f"""
                Bạn là một chuyên gia phân tích kinh tế vĩ mô hàng đầu, đang chuẩn bị một báo cáo tư vấn.
                Dưới đây là bộ dữ liệu kinh tế vĩ mô của **{country}** từ năm {selected_start_year} đến {selected_end_year}:
                
                {data_string}
                
                Dựa trên bộ dữ liệu này, hãy thực hiện phân tích chi tiết cho đối tượng là: **{audience}**.
                Cấu trúc báo cáo của bạn phải tuân thủ nghiêm ngặt 5 phần sau:

                **1. Bối cảnh & Dữ liệu chính:**
                Tóm tắt ngắn gọn bối cảnh kinh tế của {country} trong giai đoạn được cung cấp. Nêu bật các chỉ số chính và mức trung bình của chúng.

                **2. Xu hướng nổi bật & Biến động:**
                Phân tích các xu hướng tăng/giảm rõ rệt nhất. Chỉ ra những năm có biến động mạnh nhất và giải thích ngắn gọn nguyên nhân.

                **3. Tương quan đáng chú ý:**
                Chỉ ra các mối tương quan thú vị và diễn giải ý nghĩa.

                **4. Kiến nghị cho đối tượng: {audience}**
                Cung cấp 3-4 kiến nghị chiến lược, cụ thể, hữu ích. (Nếu đối tượng là "Ngân hàng Agribank", hãy tập trung kiến nghị vào bối cảnh của Việt Nam).

                **5. Hành động thực thi (kèm KPI/Điều kiện kích hoạt):**
                Từ các kiến nghị ở mục 4, đề xuất 1-2 hành động cụ thể kèm KPI hoặc "Điều kiện kích hoạt".
                
                Hãy trình bày rõ ràng, súc tích và chuyên nghiệp.
                """
                
                with st.spinner(f"AI đang phân tích {country} và tạo báo cáo cho {audience}..."):
                    response = model.generate_content(prompt_template)
                    return response.text
                    
            except Exception as e:
                if "API_key" in str(e):
                    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thiết lập trong file .streamlit/secrets.toml")
                elif "API key is invalid" in str(e):
                     st.error("Lỗi: GEMINI_API_KEY không hợp lệ. Vui lòng kiểm tra lại trong file .streamlit/secrets.toml")
                else:
                    st.error(f"Đã xảy ra lỗi khi gọi AI: {e}")
                return None

        # Nút kích hoạt AI
        if st.button(f"🚀 Sinh AI phân tích và tư vấn cho {target_audience}"):
            # *** THAY ĐỔI: Dùng df_processed_sidebar ***
            if df_processed_sidebar.empty:
                st.error("Không có dữ liệu để phân tích. Vui lòng chọn chỉ số ở thanh bên.")
            else:
                # Dùng df_processed_sidebar để AI chỉ phân tích các chỉ số đã chọn
                ai_report = generate_ai_analysis(df_processed_sidebar, selected_country, target_audience)
                if ai_report:
                    st.markdown(ai_report)
