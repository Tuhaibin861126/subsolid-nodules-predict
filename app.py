# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 设置页面配置 (必须是第一个 Streamlit 命令)
st.set_page_config(page_title="Solid Cancer Prediction", layout="wide")

# 加载资源 (现在在 set_page_config 之后)
@st.cache_resource
def load_assets():
    model = joblib.load('solidcancer_model.pkl')
    scaler = joblib.load('solidcancer_scaler.pkl')
    numerical_features = joblib.load('solidcancer_numerical_features.pkl')
    categorical_features = joblib.load('solidcancer_categorical_features.pkl')
    # 如果使用了 LabelEncoder，也加载它
    # label_encoder = joblib.load('solidcancer_label_encoder.pkl')
    return model, scaler, numerical_features, categorical_features #, label_encoder

model, scaler, numerical_features, categorical_features = load_assets() #, label_encoder = load_assets()


st.title('Solid Cancer Prediction Model')

# 输入表单
input_data = {}
col1, col2, col3, col4 = st.columns(4)  # 根据特征数量调整列数

with col1:
    input_data['rag'] = st.number_input('rag:', min_value=0.0, format="%.2f")

with col2:
    input_data['solidratio'] = st.number_input('solidratio:', min_value=0.0, format="%.2f")

with col3:
    input_data['NSE'] = st.number_input('NSE:', min_value=0.0, format="%.2f")

with col4:
    input_data['diameter'] = st.number_input('Diameter (mm):', min_value=0.0, format="%.2f")


# 预测逻辑
if st.button('Predict'):
    try:
        # 创建包含所有特征的DataFrame（顺序：数值特征在前，分类在后）
        df_numerical = pd.DataFrame([input_data])[numerical_features]
        df_categorical = pd.DataFrame([input_data])[categorical_features]  # 如果没有分类特征，这将是一个空 DataFrame

        # 标准化数值特征
        scaled_numerical = scaler.transform(df_numerical)

        # 合并数据 (如果没有分类特征，则直接使用 scaled_numerical)
        if categorical_features:
            final_input = np.concatenate([scaled_numerical, df_categorical.values], axis=1)
        else:
            final_input = scaled_numerical

        # 预测
        probability = model.predict_proba(final_input)[0, 1]

        # 显示结果
        st.success(f'Predicted Probability of Being Positive: **{probability:.1%}**')

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug Info - Numerical Features:", numerical_features)
        st.write("Debug Info - Categorical Features:", categorical_features)
