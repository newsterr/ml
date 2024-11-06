import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# จำนวนข้อมูลที่จะสร้าง (7 วัน เก็บทุก ๆ 10 นาที)
num_days = 7
interval_minutes = 10
num_samples = int((24 * 60 / interval_minutes) * num_days)  # จำนวนข้อมูลทั้งหมด

# ตั้งวันที่เริ่มต้น
start_date = datetime.now()

# สร้างข้อมูลทดสอบ
data = {
    "timestamp": [(start_date + timedelta(minutes=interval_minutes * i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(num_samples)],
    "DHT22_TEMP": np.random.normal(loc=30, scale=5, size=num_samples),          # อุณหภูมิรอบข้าง (ค่าเฉลี่ยที่ 30°C)
    "DHT22_HUM": np.random.normal(loc=50, scale=10, size=num_samples),           # ความชื้นสัมพัทธ์ (ค่าเฉลี่ยที่ 50%)
    "Voltage": np.random.normal(loc=220, scale=10, size=num_samples),            # แรงดันไฟฟ้า (ค่าเฉลี่ยที่ 220V)
    "Current": np.random.normal(loc=5, scale=1, size=num_samples),               # กระแสไฟฟ้า (ค่าเฉลี่ยที่ 5A)
    "Temp_PV": np.random.normal(loc=40, scale=5, size=num_samples),              # อุณหภูมิของแผงโซล่าเซลล์ (ค่าเฉลี่ยที่ 40°C)
    "Irradiance": [
        0 if (hour < 6 or hour > 18) else np.random.normal(loc=800, scale=100) * (1 - abs(12 - hour) / 6)
        for i in range(num_samples)
        for hour in [(start_date + timedelta(minutes=interval_minutes * i)).hour]
    ],  # ปริมาณแสงอาทิตย์ตามช่วงเวลา (เช้าและเย็นต่ำ, กลางวันสูง)
    "Dust Density": np.random.normal(loc=50, scale=20, size=num_samples),  # หน่วยเป็น µg/m³
    "Wind": np.random.normal(loc=10, scale=3, size=num_samples),                 # ความเร็วลม (ค่าเฉลี่ยที่ 10 m/s)
}

# คำนวณ Power (พลังงานที่ผลิตได้) = Voltage * Current * Efficiency Factor
# Efficiency Factor ขึ้นอยู่กับ Irradiance

data["Power"] = [
    data["Voltage"][i] * data["Current"][i] * (data["Irradiance"][i] / 1000) if data["Irradiance"][i] > 0 else 0
    for i in range(num_samples)
]

# สร้าง DataFrame จากข้อมูลที่จำลอง
df = pd.DataFrame(data)

# แสดงข้อมูลบางส่วน
print(df.head())

# บันทึกข้อมูลลง CSV เพื่อใช้ในการทดสอบ
df.to_csv('solar_7days_test_data.csv', index=False)

print("สร้างข้อมูลทดสอบสำเร็จ และบันทึกลงในไฟล์ 'solar_7days_test_data.csv'")
