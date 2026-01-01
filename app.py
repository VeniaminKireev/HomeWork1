import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Прогнозирование цен на автомобили",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_features():
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return features

@st.cache_data
def load_data():
    train_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    return pd.read_csv(train_url)

st.title("🚗 Прогнозирование цен на автомобили")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Предсказание", "📈 Визуализация модели"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    data = load_data()
    
    st.subheader("Основная информация о данных")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Количество строк", data.shape[0])
        st.metric("Количество признаков", data.shape[1])
    
    with col2:
        st.metric("Пропущенные значения", data.isnull().sum().sum())
        st.metric("Типы данных", len(data.dtypes.unique()))
    
        st.subheader("Визуализация распределений")
    selected_feature = st.selectbox(
        "Выберите признак для анализа:",
        data.select_dtypes(include=[np.number]).columns.tolist()
    )
    
    if selected_feature:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(data[selected_feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title(f"Распределение {selected_feature}")
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel("Частота")
        
        ax2.boxplot(data[selected_feature].dropna())
        ax2.set_title(f"Boxplot {selected_feature}")
        ax2.set_ylabel(selected_feature)
        
        st.pyplot(fig)
    
    st.subheader("Матрица корреляций")
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title("Матрица корреляций числовых признаков")
        st.pyplot(fig)

with tab2:
    st.header("Предсказание цены автомобиля")
    
    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["📁 Загрузить CSV файл", "✍️ Ручной ввод"]
    )
    
    if input_method == "📁 Загрузить CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.success("Файл успешно загружен!")
                st.write("Предпросмотр данных:")
                st.dataframe(input_data.head())
                
                              
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {e}")
    
    else:  
        st.subheader("Введите параметры автомобиля:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Год выпуска", min_value=1900, max_value=2023, value=2015)
            km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
            mileage = st.number_input("Расход топлива", min_value=0.0, value=20.0)
            engine = st.number_input("Объем двигателя (CC)", min_value=0, value=1500)
        
        with col2:
            max_power = st.number_input("Максимальная мощность", min_value=0.0, value=100.0)
            seats = st.number_input("Количество сидений", min_value=2, max_value=10, value=5)
            fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "CNG", "LPG"])
            transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
        
        if st.button("Предсказать цену", type="primary"):
            input_dict = {
                'year': year,
                'km_driven': km_driven,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats,
                'fuel': fuel,
                'transmission': transmission
            }
            
                     
            st.info("Функционал предсказания требует дополнительной настройки трансформеров")

with tab3:
    st.header("Визуализация модели")
    
    try:
        model = load_model()
        features = load_features()
        st.subheader("Важность признаков в модели")
        
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            coef_df = pd.DataFrame({
                'Признак': features[:len(coefficients)],  # Обрезаем, если нужно
                'Коэффициент': coefficients,
                'Абсолютное значение': np.abs(coefficients)
            })
            
            coef_df = coef_df.sort_values('Абсолютное значение', ascending=False)
            
            top_n = min(20, len(coef_df))
            top_coef = coef_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['red' if x < 0 else 'green' for x in top_coef['Коэффициент']]
            ax.barh(top_coef['Признак'], top_coef['Коэффициент'], color=colors)
            ax.set_xlabel('Коэффициент')
            ax.set_title(f'Топ-{top_n} самых важных признаков')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
            
            st.subheader("Таблица коэффициентов")
            st.dataframe(coef_df[['Признак', 'Коэффициент']].sort_values('Коэффициент', ascending=False))
            
        else:
            st.warning("У модели нет атрибута coef_ для визуализации")
            
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.info("Убедитесь, что файлы model.pkl и feature_names.pkl находятся в той же директории")

st.markdown("---")
st.markdown("### 📝 Примечания")
st.info("""
1. Для полноценной работы приложения необходимо:
   - Сохранить все трансформеры (OneHotEncoder, StandardScaler)
   - Реализовать полный пайплайн предобработки
2. EDA выполняется на загруженных данных
3. Модель показывает важность признаков через коэффициенты
""")
