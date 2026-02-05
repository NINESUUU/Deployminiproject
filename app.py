import streamlit as st
import pandas as pd
import joblib
import re

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Job Classifier Pro",
    layout="wide"
)

st.title("🤖 Job Classifier (Default Model)")

# ==============================
# Load Default Model
# ==============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("matplotlib_model1.joblib")
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

st.success("✅ ใช้งานโมเดล: matplotlib_model1.joblib")

# ==============================
# Helper: Extract Info from Text
# ==============================
def extract_from_text(text):
    data = {}
    text_lower = text.lower()

    tech_kw = ['python', 'java', 'react', 'node', 'aws', 'api', 'backend', 'frontend']
    data_kw = ['sql', 'data', 'analysis', 'tableau', 'powerbi', 'machine learning', 'ai']

    data['tech'] = min(sum(1 for w in tech_kw if w in text_lower) * 20 + 10, 90)
    data['data'] = min(sum(1 for w in data_kw if w in text_lower) * 20 + 10, 90)

    salary_match = re.search(r'(\d{2,3})[,.]?000', text)
    if salary_match:
        data['salary'] = int(salary_match.group(1)) * 1000

    exp_match = re.search(r'(\d+)\s*(year|yr|ปี)', text_lower)
    if exp_match:
        data['exp'] = int(exp_match.group(1))

    data['remote'] = 1 if 'remote' in text_lower or 'work from home' in text_lower else 0
    return data

# ==============================
# UI: Text Input
# ==============================
st.subheader("1️⃣ ใส่ Job Description")
jd_text = st.text_area(
    "วางรายละเอียดงานที่นี่ (ระบบช่วยเดาค่าให้)",
    height=120
)

defaults = {
    'tech': 50, 'data': 10, 'design': 5,
    'sales': 5, 'mkt': 5, 'ops': 5,
    'salary': 30000, 'exp': 2, 'remote': 0
}

if jd_text:
    defaults.update(extract_from_text(jd_text))
    st.caption(
        f"💡 Auto-fill → Tech: {defaults['tech']} | "
        f"Salary: {defaults.get('salary', '?')} | "
        f"Exp: {defaults.get('exp', '?')} ปี"
    )

st.divider()

# ==============================
# UI: Manual Input
# ==============================
st.subheader("2️⃣ ตรวจสอบ / แก้ไขข้อมูล")

with st.form("main_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🛠️ Skills")
        tech = st.number_input("Tech", 0, 100, defaults['tech'])
        data = st.number_input("Data", 0, 100, defaults['data'])
        design = st.number_input("Design", 0, 100, defaults['design'])
        sales = st.number_input("Sales", 0, 100, defaults['sales'])
        mkt = st.number_input("Marketing", 0, 100, defaults['mkt'])
        ops = st.number_input("Operations", 0, 100, defaults['ops'])

    with c2:
        st.markdown("### 💼 Position Info")
        seniority = st.selectbox(
            "Seniority", [0,1,2,3,4],
            format_func=lambda x: ["Intern","Junior","Mid","Senior","Lead"][x],
            index=2
        )
        contract = st.selectbox(
            "Contract", [0,1,2],
            format_func=lambda x: ["Full-time","Contract","Intern"][x]
        )
        edu = st.selectbox(
            "Education", [0,1,2,3],
            format_func=lambda x: ["Any","Bachelor","Master","PhD"][x],
            index=1
        )
        lang = st.selectbox(
            "Language", [0,1,2],
            format_func=lambda x: ["Local","English","Bilingual"][x]
        )

    with c3:
        st.markdown("### 💰 Other")
        salary = st.number_input("Salary (บาท)", 0, 500000, defaults['salary'], step=1000)
        exp = st.number_input("Experience (ปี)", 0, 20, defaults['exp'])
        remote = st.selectbox("Remote?", [0,1], format_func=lambda x: "Yes" if x else "No")
        req_cnt = st.number_input("Requirements Count", 1, 50, 5)
        resp_cnt = st.number_input("Responsibilities Count", 1, 50, 5)

    submit = st.form_submit_button("🔮 Predict", type="primary")

# ==============================
# Prediction
# ==============================
if submit:
    input_df = pd.DataFrame([{
        'tech_skill': tech,
        'data_skill': data,
        'design_skill': design,
        'sales_skill': sales,
        'marketing_skill': mkt,
        'ops_skill': ops,
        'seniority': seniority,
        'contract_type': contract,
        'edu_min': edu,
        'lang_req': lang,
        'min_years_exp': exp,
        'salary_min': salary,
        'remote_flag': remote,
        'requirements_count': req_cnt,
        'responsibilities_count': resp_cnt,

        # ===== features ที่โมเดลต้องการ =====
        'salary_max': salary * 1.2,
        'salary_per_year_exp': salary / max(exp, 1),
        'req_to_resp_ratio': req_cnt / max(resp_cnt, 1),
        'skill_density': (tech + data + design + sales + mkt + ops) / 600,
        'tools_mentioned': 0,
        'region_code': 0,
        'complexity': seniority
    }])

    job_map = {
        0: "Software Engineer",
        1: "Data",
        2: "Design",
        3: "Sales",
        4: "Marketing",
        5: "Operations"
    }

    pred = model.predict(input_df)[0]

    st.divider()
    st.subheader("3️⃣ ผลลัพธ์")
    st.success(f"ตำแหน่งที่เหมาะสม: **{job_map.get(pred, 'Unknown')}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame(
            probs,
            index=[job_map.get(i, str(i)) for i in range(len(probs))],
            columns=["Probability"]
        )
        st.bar_chart(prob_df)
