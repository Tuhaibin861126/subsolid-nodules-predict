# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 设置页面配置 (必须是第一个 Streamlit 命令)
st.set_page_config(page_title="Lung Subsolid Nodules Prediction", layout="wide")

# 加载资源 (现在在 set_page_config 之后)
@st.cache_resource
def load_assets():
    model = joblib.load('solidcancer_model.pkl')
    scaler = joblib.load('solidcancer_scaler.pkl')
    numerical_features = joblib.load('solidcancer_numerical_features.pkl')
    categorical_features = joblib.load('solidcancer_categorical_features.pkl')
    label_encoder = joblib.load('solidcancer_label_encoder.pkl')  # 加载 LabelEncoder
    return model, scaler, numerical_features, categorical_features, label_encoder

model, scaler, numerical_features, categorical_features, label_encoder = load_assets()


st.title('Lung Subsolid Nodules Prediction Model')

# 输入表单
input_data = {}
col1, col2, col3, col4 = st.columns(4)  # 根据特征数量调整列数

with col1:
    input_data['rag'] = st.selectbox('Rag:', options=[0, 1])  # 使用 selectbox，因为 rag 是二分类

with col2:
    input_data['solidratio'] = st.number_input('Solidratio:', min_value=0.0, format="%.2f")

with col3:
    input_data['CYFRA 21-1'] = st.number_input('CYFRA 21-1:', min_value=0.0, format="%.2f")

with col4:
    input_data['diameter'] = st.number_input('Diameter (mm):', min_value=0.0, format="%.2f")


# 预测逻辑
if st.button('Predict'):
    try:
        # 创建包含所有特征的DataFrame（顺序：数值特征在前，分类在后）
        df = pd.DataFrame([input_data])
        df_numerical = df[numerical_features]
        df_categorical = df[categorical_features]

        # 标准化数值特征
        scaled_numerical = scaler.transform(df_numerical)

        # 对分类特征进行编码
        encoded_categorical = label_encoder.transform(df_categorical['rag'])
        encoded_categorical = encoded_categorical.reshape(1, -1)  # 转换为二维数组

        # 合并数据
        final_input = np.concatenate([scaled_numerical, encoded_categorical], axis=1)

        # 预测
        probability = model.predict_proba(final_input)[0, 1]

        # 显示结果
        st.success(f'Predicted Probability of Malignancy: **{probability:.1%}**')

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug Info - Numerical Features:", numerical_features)
        st.write("Debug Info - Categorical Features:", categorical_features)
