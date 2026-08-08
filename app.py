import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="پیش‌بینی قیمت آپارتمان", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("price_model_artifacts.pkl")

art = load_artifacts()

st.title("🏠 پیش‌بینی قیمت آپارتمان")
st.caption("بر اساس مدل CatBoost آموزش‌دیده روی داده‌های دیوار")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("شهر", options=art['known_cities'])
    # فقط محله‌های همان شهر را نشان بده
    neigh_options = [n for n in art['known_neighborhoods'] if n.startswith(city + '_')]
    neigh_display = [n.replace(city + '_', '') for n in neigh_options] or ['unknown']
    neighborhood_display = st.selectbox("محله", options=neigh_display)
    city_neighborhood = f"{city}_{neighborhood_display}"

    building_size = st.number_input("متراژ (متر مربع)", min_value=15, max_value=500, value=80)
    rooms_count = st.selectbox("تعداد اتاق", options=[0, 1, 2, 3, 4, 5], index=2)

with col2:
    floor = st.number_input("طبقه", min_value=0, max_value=30, value=2)
    building_age = st.number_input("سن بنا (سال)", min_value=0, max_value=80, value=5)

    deed_type = st.selectbox("نوع سند", options=["تک برگ", "قولنامه‌ای", "منگوله‌دار", "سایر"])
    floor_material = st.selectbox("جنس کف", options=["سرامیک", "پارکت", "موزاییک", "سایر"])

st.markdown("**امکانات**")
c1, c2, c3, c4 = st.columns(4)
has_elevator = c1.checkbox("آسانسور", value=True)
has_parking = c2.checkbox("پارکینگ", value=True)
has_warehouse = c3.checkbox("انباری", value=True)
has_balcony = c4.checkbox("بالکن", value=False)

c5, c6, c7, c8 = st.columns(4)
has_restroom = c5.checkbox("سرویس بهداشتی", value=True)
has_heating_system = c6.checkbox("سیستم گرمایشی", value=True)
has_cooling_system = c7.checkbox("سیستم سرمایشی", value=True)
is_rebuilt = c8.checkbox("بازسازی‌شده", value=False)


def build_feature_row():
    row = {}

    # ویژگی‌های پایه
    row['rooms_count'] = rooms_count
    row['floor'] = floor
    row['building_age'] = building_age
    row['building_size'] = building_size

    # ویژگی‌های missing (چون کاربر همیشه پر می‌کند، صفر)
    row['rooms_count_missing'] = 0
    row['floor_missing'] = 0
    row['building_age_missing'] = 0

    # مهندسی ویژگی (دقیقاً هم‌راستا با آموزش)
    row['log_building_size'] = np.log1p(building_size)
    row['building_size_squared'] = building_size ** 2
    row['age_squared'] = building_age ** 2
    row['is_new'] = int(building_age <= 3)
    row['price_per_room'] = building_size / (rooms_count + 1)
    row['is_ground_floor'] = int(floor == 0)
    row['is_top_floor'] = int(floor >= 5)

    # target encoding -- با fallback به global_mean اگر شهر/محله جدید بود
    row['city_target_enc'] = art['city_means'].get(city, art['global_mean'])
    row['city_neighborhood_target_enc'] = art['city_neigh_means'].get(city_neighborhood, art['global_mean'])

    # categorical (رشته‌ای، دقیقاً هم‌فرمت با آموزش: lower/strip)
    cat_map = {
        'deed_type': deed_type,
        'has_balcony': 'yes' if has_balcony else 'no',
        'has_elevator': 'yes' if has_elevator else 'no',
        'has_warehouse': 'yes' if has_warehouse else 'no',
        'has_parking': 'yes' if has_parking else 'no',
        'is_rebuilt': 'yes' if is_rebuilt else 'no',
        'has_warm_water_provider': 'unknown',
        'has_heating_system': 'yes' if has_heating_system else 'no',
        'has_cooling_system': 'yes' if has_cooling_system else 'no',
        'has_restroom': 'yes' if has_restroom else 'no',
        'building_direction': 'unknown',
        'floor_material': floor_material,
    }
    for k, v in cat_map.items():
        if k in art['categorical_features']:
            row[k] = str(v).strip().lower()

    df_row = pd.DataFrame([row])

    # هم‌ترازسازی دقیق ستون‌ها با ترتیب train؛ هر ستون جا‌مانده با NaN/unknown پر می‌شود
    for col in art['feature_columns']:
        if col not in df_row.columns:
            df_row[col] = 'unknown' if col in art['categorical_features'] else 0
    df_row = df_row[art['feature_columns']]
    return df_row


if st.button("پیش‌بینی قیمت 💰", type="primary", use_container_width=True):
    X_input = build_feature_row()
    preds_log = np.mean([m.predict(X_input) for m in art['models']], axis=0)
    pred_price = np.expm1(preds_log)[0]

    st.success(f"### قیمت پیش‌بینی‌شده: {pred_price:,.0f} تومان")
    st.caption(
        f"معادل تقریبی {pred_price/building_size:,.0f} تومان به‌ازای هر متر مربع. "
        "این یک تخمین آماری است، نه قیمت قطعی؛ خطای متوسط مدل حدود ۲۴٪ است."
    )
