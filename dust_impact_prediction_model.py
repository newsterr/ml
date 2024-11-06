import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ขั้นตอนที่ 1: โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('solar_7days_test_data.csv')

# ตรวจสอบข้อมูลเบื้องต้น
print(df.head())
print(df.info())

# แปลง timestamp เป็น datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ขั้นตอนที่ 2: สร้าง Dust Impact Prediction Model (Random Forest Regression)
# เตรียมข้อมูลสำหรับการทำนายผลกระทบของฝุ่น
X = df[['Dust Density', 'Irradiance', 'Temp_PV', 'DHT22_TEMP', 'DHT22_HUM', 'Wind']]
y = df['Power']

# แบ่งข้อมูลเป็น train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ทำนายและประเมินผล
y_pred = rf_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# คำนวณเกณฑ์ของ Dust Density ที่ควรทำความสะอาด
# ใช้ค่าเฉลี่ย + 2 เท่าของส่วนเบี่ยงเบนมาตรฐานเป็นเกณฑ์ในการแจ้งเตือน
mean_dust_density = df['Dust Density'].mean()
std_dust_density = df['Dust Density'].std()
threshold_dust_density = mean_dust_density + 2 * std_dust_density
print(f"เกณฑ์ปริมาณฝุ่นที่ควรทำความสะอาด: {threshold_dust_density:.2f} µg/m³")

# ทำเครื่องหมายข้อมูลที่เกินเกณฑ์
df['needs_cleaning'] = df['Dust Density'] > threshold_dust_density

# เพิ่มข้อมูลการทำนายพลังงาน (Predicted Power) และ anomaly ลงใน DataFrame
result_df = X_test.copy()
result_df['Actual Power'] = y_test
result_df['Predicted Power'] = y_pred
result_df['timestamp'] = df['timestamp'].loc[result_df.index]
result_df['Dust Density'] = df['Dust Density'].loc[result_df.index]

# สร้างโมเดล Isolation Forest สำหรับการตรวจหาความผิดปกติ
from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
isolation_forest.fit(X)
df['anomaly'] = isolation_forest.predict(X)
result_df['anomaly'] = df['anomaly'].loc[result_df.index]

# Visualization การแสดงผลการทำนายพลังงานและ Dust Density ด้วย Plotly
fig = px.scatter(result_df, x='timestamp', y=['Actual Power', 'Predicted Power', 'Dust Density'],
                 title='Actual vs Predicted Power and Dust Density over Time',
                 labels={'value': 'Power', 'timestamp': 'Time', 'variable': 'Legend'},
                 template='plotly_white')
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Power',
    legend_title='Legend'
)
fig.update_traces(mode='markers')
fig.show()

# เพิ่มการแสดงผลจุดที่ควรทำความสะอาดตามเกณฑ์ของ Dust Density
cleaning_points = df[df['needs_cleaning']]
fig_cleaning = go.Figure()
fig_cleaning.add_trace(go.Scatter(x=cleaning_points['timestamp'], y=cleaning_points['Dust Density'],
                                  mode='markers', name='Needs Cleaning (Dust Density)',
                                  marker=dict(color='green', size=7)))

# ปรับแต่ง layout
fig_cleaning.update_layout(
    title='Dust Density Cleaning Threshold over Time',
    xaxis_title='Time',
    yaxis_title='Dust Density (µg/m³)',
    legend_title='Features',
    template='plotly_white',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

fig_cleaning.show()

# Visualization การแสดงผลการทำนายพลังงาน, Dust Density และข้อมูลความผิดปกติในกราฟเดียว
fig_combined = go.Figure()

# แสดง Actual Power
fig_combined.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['Actual Power'],
                                  mode='markers', name='Actual Power',
                                  marker=dict(size=5, color='blue', opacity=0.6)))

# แสดง Predicted Power
fig_combined.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['Predicted Power'],
                                  mode='markers', name='Predicted Power',
                                  marker=dict(size=5, color='orange', opacity=0.6)))

# แสดง Dust Density
fig_combined.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['Dust Density'],
                                  mode='markers', name='Dust Density',
                                  marker=dict(size=5, color='green', opacity=0.6)))

# แสดงความผิดปกติ (Anomalies)
anomalies = result_df[result_df['anomaly'] == -1]
fig_combined.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['Dust Density'],
                                  mode='markers', name='Anomalies (Dust Density)',
                                  marker=dict(color='red', size=7)))

# ปรับแต่ง layout
fig_combined.update_layout(
    title='Combined Features: Actual vs Predicted Power, Dust Density, and Anomalies over Time',
    xaxis_title='Time',
    yaxis_title='Values',
    legend_title='Features',
    template='plotly_white',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

fig_combined.show()
