import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 กำลังเทรนโมเดลเวอร์ชัน Full Option...")

# 1. โหลดข้อมูล
try:
    df = pd.read_csv('15.job_description_synthetic.csv')
except:
    print("❌ ไม่พบไฟล์ CSV")
    exit()

# 2. เลือก Features ให้ครบตามโจทย์ PDF (หน้า 10-11)
features = [
    # กลุ่ม Skill Intensity (0-100)
    'tech_skill', 'data_skill', 'design_skill', 
    'sales_skill', 'marketing_skill', 'ops_skill',
    # กลุ่มข้อมูลงาน (Categorical Codes)
    'seniority',       # 0=Intern, 1=Junior, 2=Mid, 3=Senior, 4=Lead
    'contract_type',   # 0=Fulltime, 1=Contract, 2=Internship
    'edu_min',         # 0=Any, 1=Bachelor, 2=Master, 3=PhD
    'lang_req',        # 0=Local, 1=English, 2=Bilingual
    # กลุ่มตัวเลข (Numeric)
    'min_years_exp', 
    'salary_min', 
    'remote_flag',
    'requirements_count',   # จำนวนข้อ Req
    'responsibilities_count' # จำนวนข้อ Resp
]

X = df[features]
y = df['job_family']

# 3. เตรียมข้อมูล (Split & Scale)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. สร้าง 2 โมเดลเพื่อเปรียบเทียบ
print("🤖 Training Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
print(f"   👉 RF Score: {model_rf.score(X_test, y_test)*100:.2f}%")

print("🤖 Training Logistic Regression...")
model_lr = LogisticRegression(max_iter=2000)
model_lr.fit(X_train_scaled, y_train)
print(f"   👉 LR Score: {model_lr.score(X_test_scaled, y_test)*100:.2f}%")

# 5. บันทึกทุกอย่าง
joblib.dump(model_rf, 'model_jd_rf.pkl')
joblib.dump(model_lr, 'model_jd_lr.pkl')
joblib.dump(scaler, 'scaler_jd.pkl')
print("✅ บันทึกไฟล์โมเดลชุดใหม่เรียบร้อย!")