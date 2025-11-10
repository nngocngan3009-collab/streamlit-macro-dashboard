import re

@st.cache_data(show_spinner=False, ttl=1200)
def search_indicators(keyword: str, top: int = 25) -> pd.DataFrame:
    """
    Tìm indicator cho WB_WDI:
      - Data360: chỉ nhận idno bắt đầu bằng 'WB_WDI_'
      - Chuẩn hoá: WB_WDI_* -> pretty_id = mã chấm (NY.GDP.MKTP.CD)
      - Nếu Data360 lỗi/không có item hợp lệ -> fallback World Bank catalog
    Trả DF: columns = [name, full_id, wb_dot_id, pretty_id]
    """
    valid = []

    # ---- 1) Thử Data360 (chỉ nhận WB_WDI)
    try:
        body = {
            "count": True,
            "select": "series_description/idno, series_description/name, series_description/database_id",
            "search": keyword,
            "top": int(top),
            "filter": "series_description/database_id eq 'WB_WDI' and type eq 'indicator'"
        }
        raw = http_post_json(f"{DATA360_BASE_URL}{D360_SEARCH_ENDPOINT}", body)

        rows = raw.get("value") or raw.get("items") or raw
        if isinstance(rows, dict):
            rows = rows.get("value") or rows.get("items") or []

        for r in rows:
            sd = r.get("series_description") if isinstance(r.get("series_description"), dict) else {}
            idno = r.get("series_description/idno") or sd.get("idno")
            name = r.get("series_description/name") or sd.get("name") or idno

            # Chỉ nhận idno đúng chuẩn WB_WDI
            if not idno or not idno.startswith("WB_WDI_"):
                continue

            core = idno[len("WB_WDI_"):]            # NY_GDP_MKTP_CD
            wb_dot = core.replace("_", ".")          # NY.GDP.MKTP.CD

            valid.append({
                "name": name,
                "full_id": idno,
                "wb_dot_id": wb_dot,
                "pretty_id": wb_dot
            })
    except Exception:
        pass

    # ---- 2) Nếu có item hợp lệ từ Data360 -> trả luôn
    if valid:
        return pd.DataFrame(valid).head(int(top))

    # ---- 3) Fallback: World Bank catalog (đảm bảo 'GDP' luôn ra)
    base = f"{WB_BASE}/indicator"
    per_page = 20000
    url = f"{base}?format=json&per_page={per_page}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()
    items = payload[1] if isinstance(payload, list) and len(payload) > 1 else []

    # lọc theo keyword ở name hoặc id
    k = (keyword or "").lower()
    rows = []
    for it in items:
        iid = it.get("id")         # ví dụ: NY.GDP.MKTP.CD
        name = it.get("name") or iid
        if not iid:
            continue
        if k and (k not in (name or "").lower() and k not in iid.lower()):
            continue

        # Chỉ nhận ID đúng format WDI (A.B.C.D… chữ + chấm)
        if not re.match(r"^[A-Z]{2}\.[A-Z0-9]+\.[A-Z0-9.]+$", iid):
            continue

        rows.append({
            "name": name,
            "full_id": None,          # không có từ catalog
            "wb_dot_id": iid,         # dùng WB v2 trực tiếp
            "pretty_id": iid
        })

    return pd.DataFrame(rows).head(int(top))
