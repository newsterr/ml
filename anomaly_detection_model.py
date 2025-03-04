#Isolation Forest
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# ขั้นตอนที่ 1: โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('solar_7days_test_data.csv')

# ตรวจสอบข้อมูลเบื้องต้น
print(df.head())
print(df.info())

# แปลง timestamp เป็น datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ขั้นตอนที่ 2: สร้าง Anomaly Detection Model (Isolation Forest)
# เตรียมข้อมูลสำหรับ Anomaly Detection
features = ['DHT22_TEMP', 'DHT22_HUM', 'Voltage', 'Current', 'Temp_PV', 'Irradiance', 'Dust Density', 'Wind']
X = df[features]

# สร้างและฝึกโมเดล Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
isolation_forest.fit(X)

# ทำนายความผิดปกติ (-1 = anomalous, 1 = normal)
df['anomaly'] = isolation_forest.predict(X)

# แสดงข้อมูลที่เป็นความผิดปกติ
anomalies = df[df['anomaly'] == -1]
print(f"พบ {len(anomalies)} ความผิดปกติในข้อมูล")

# Visualization การแสดงความผิดปกติด้วย Plotly สำหรับการเลื่อนปรับเพื่อดูรายละเอียดข้อมูล
features_to_plot = ['DHT22_TEMP', 'DHT22_HUM', 'Voltage', 'Current', 'Temp_PV', 'Irradiance', 'Dust Density', 'Wind']

for feature in features_to_plot:
    fig = px.scatter(df, x='timestamp', y=feature, color='anomaly',
                     title=f'Anomalies in {feature} over Time',
                     labels={
                         'timestamp': 'Time',
                         feature: feature,
                         'anomaly': 'Anomaly Status'
                     },
                     color_discrete_map={-1: 'red', 1: 'blue'})
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

# เพิ่มกราฟที่รวมทุกฟีเจอร์เพื่อดูภาพรวม
fig_combined = go.Figure()

for feature in features_to_plot:
    fig_combined.add_trace(go.Scatter(x=df['timestamp'], y=df[feature],
                                      mode='markers', name=feature,
                                      marker=dict(size=5, color='blue', opacity=0.6)))

# เพิ่มการแสดงผล anomaly ในกราฟรวม โดยแยกฟีเจอร์แต่ละตัว
for feature in features_to_plot:
    fig_combined.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies[feature],
                                      mode='markers', name=f'Anomalies in {feature}',
                                      marker=dict(color='red', size=7)))

# ปรับแต่ง layout
fig_combined.update_layout(
    title='Combined Features and Anomalies over Time',
    xaxis_title='Time',
    yaxis_title='Feature Values',
    legend_title='Features',
    template='plotly_white',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

fig_combined.show()
