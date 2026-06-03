import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="CS:GO Bomb Planting Predictor", layout="wide")

st.title("🎮 Предсказание установки бомбы в CS:GO")
st.markdown("---")

@st.cache_resource
def load_all_models():
    scaler = joblib.load('models/scaler_class.pkl')
    selector = joblib.load('models/selector_class.pkl')
    
    with open('models/ml1_logreg.pkl', 'rb') as f:
        logreg = pickle.load(f)
    
    lgbm = lgb.Booster(model_file='models/ml2_lightgbm.txt')
    
    cat = CatBoostClassifier()
    cat.load_model('models/ml3_catboost.cbm')
    
    with open('models/ml4_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    
    with open('models/ml5_stacking.pkl', 'rb') as f:
        stacking = pickle.load(f)
    
    keras_model = load_model('models/ml6_keras_model.h5')
    
    return scaler, selector, logreg, lgbm, cat, rf, stacking, keras_model

scaler, selector, logreg, lgbm, cat, rf, stacking, keras_model = load_all_models()

feature_names = [
'time_left', 'ct_score', 't_score', 'ct_health', 't_health', 'ct_armor', 't_armor',
 'ct_money', 't_money', 'ct_helmets', 't_helmets', 'ct_defuse_kits', 'ct_players_alive',
   't_players_alive', 'total_health', 'score_difference', 'total_money', 'map_de_cache', 
   'map_de_dust2', 'map_de_inferno', 'map_de_mirage', 'map_de_nuke', 'map_de_overpass', 'map_de_train',
     'map_de_vertigo', 'leading_team_CT', 'leading_team_T', 'leading_team_Tie'
]

st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу",
    ["О разработчике", "О данных", "Визуализации", "Предсказание"]
)

if page == "О разработчике":
    st.header("Разработчик")
    st.write("**ФИО:** [Введите ваше ФИО]")
    st.write("**Группа:** [Введите группу]")
    st.image("ргр.jpg", width=200, caption="Фото")  
    st.write("**Тема РГР:** Разработка Web-приложения для инференса ML моделей и анализа данных (CS:GO Bomb Planting Prediction)")

elif page == "О данных":
    st.header("Набор данных: CS:GO Round Statistics")
    st.markdown("""
    **Предметная область:** Киберспорт (игра Counter-Strike: Global Offensive).
    
    **Целевая переменная:** `bomb_planted` – была ли заложена бомба командой террористов в раунде (1 – да, 0 – нет).
    
    **Признаки (всего 28):**
    - Время, счёт, здоровье, броня, деньги, количество живых игроков, наличие шлемов, наборов для разминирования.
    - Производные признаки: `total_health`, `score_difference`, `total_money`.
    - One‑hot кодированные карты (`map_de_*` – 9 признаков) и лидирующая команда (`leading_team_*` – 3 признака).
    
    **Предобработка:**
    - Заполнение пропусков, удаление выбросов по IQR.
    - Стандартизация числовых признаков (`StandardScaler`).
    - Отбор признаков (`SelectKBest`, k=14) для ускорения и снижения шума.
    """)

elif page == "Визуализации":
    st.header("Исследовательский анализ данных (EDA)")
    df = pd.read_csv('csochka.csv')
    
    st.subheader("1. Распределение целевой переменной")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df['bomb_planted'].value_counts())
    with col2:
        st.write("**Интерпретация:** резкий дисбаланс – раундов без закладки бомбы значительно больше.")
    
    st.subheader("2. Корреляционная матрица числовых признаков")
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Корреляция'})
    ax.set_title('Корреляционная матрица')
    st.pyplot(fig)
    st.caption("Насыщенные красные/синие клетки указывают на сильную положительную/отрицательную корреляцию.")
    
    st.subheader("3. Зависимость общего здоровья от оставшегося времени")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.regplot(x='time_left', y='total_health', data=df, lowess=True, 
                scatter_kws={'alpha':0.1, 's':1}, line_kws={'color':'red', 'lw':2}, ax=ax2)
    ax2.set_xlabel('Оставшееся время (сек)')
    ax2.set_ylabel('Суммарное здоровье')
    ax2.set_title('Общее здоровье vs время раунда (сглаженный тренд)')
    st.pyplot(fig2)
    st.write("**Интерпретация:** в начале раунда (большое время) здоровье максимально ≈1000, к концу раунда оно снижается. Красная линия показывает плавную тенденцию.")
    
    st.subheader("4. Средние деньги команд по картам")
    map_columns = [col for col in df.columns if col.startswith('map_de_')]
    def get_map(row):
        for col in map_columns:
            if row[col] == 1:
                return col.replace('map_de_', '')
        return 'unknown'
    df['map_name'] = df.apply(get_map, axis=1)
    
    map_money = df.groupby('map_name')[['ct_money', 't_money']].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(map_money))
    width = 0.35
    ax3.bar(x - width/2, map_money['ct_money'], width, label='CT деньги', color='blue', alpha=0.7)
    ax3.bar(x + width/2, map_money['t_money'], width, label='T деньги', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(map_money['map_name'], rotation=45, ha='right')
    ax3.set_ylabel('Среднее количество денег')
    ax3.set_title('Средние деньги команд по картам')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig3)
    st.write("**Интерпретация:** на большинстве карт деньги команд сопоставимы, но на cache  наблюдается разрыв.")

else:
    st.header("🔮 Предсказание установки бомбы")
    st.markdown("Введите параметры игрового раунда и выберите модель.")
    
    model_choice = st.selectbox(
        "Выберите модель ML",
        [
            "ML1: Logistic Regression",
            "ML2: LightGBM",
            "ML3: CatBoost",
            "ML4: Random Forest",
            "ML5: Stacking",
            "ML6: Neural Network (Keras/TF)"
        ]
    )
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_left = st.number_input("Оставшееся время (сек)", 0, 175, 100)
            ct_score = st.number_input("Счёт CT", 0, 30, 0)
            t_score = st.number_input("Счёт T", 0, 30, 0)
            ct_health = st.number_input("Здоровье CT (суммарно)", 0, 500, 500)
            t_health = st.number_input("Здоровье T (суммарно)", 0, 500, 500)
            ct_armor = st.number_input("Броня CT (суммарно)", 0, 500, 0)
            t_armor = st.number_input("Броня T (суммарно)", 0, 500, 0)
            ct_money = st.number_input("Деньги CT", 0, 50000, 4000)
            t_money = st.number_input("Деньги T", 0, 50000, 4000)
        with col2:
            ct_helmets = st.selectbox("Шлемы у CT", [0,1,2,3,4,5], index=0)
            t_helmets = st.selectbox("Шлемы у T", [0,1,2,3,4,5], index=0)
            ct_defuse_kits = st.selectbox("Наборы для разминирования", [0,1,2,3,4,5], index=0)
            ct_players_alive = st.slider("Живых игроков CT", 0, 5, 5)
            t_players_alive = st.slider("Живых игроков T", 0, 5, 5)
            total_health = ct_health + t_health
            score_difference = ct_score - t_score
            total_money = ct_money + t_money
        
        map_list = ['de_cache', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo']
        selected_map = st.selectbox("Карта", map_list)
        
        leading_team_choice = st.selectbox("Лидирующая команда", ['CT', 'T', 'Tie'])
        
        submitted = st.form_submit_button("Предсказать")
    
    if submitted:
        numeric_values = [
            time_left, ct_score, t_score, ct_health, t_health,
            ct_armor, t_armor, ct_money, t_money, ct_helmets,
            t_helmets, ct_defuse_kits, ct_players_alive, t_players_alive,
            total_health, score_difference, total_money
        ]
        
        map_onehot = [1 if selected_map == m else 0 for m in map_list]
        
        lead_onehot = [
            1 if leading_team_choice == 'CT' else 0,
            1 if leading_team_choice == 'T' else 0,
            1 if leading_team_choice == 'Tie' else 0
        ]
        
        full_input = numeric_values + map_onehot + lead_onehot
        input_df = pd.DataFrame([full_input], columns=feature_names)
        
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)
        
        if model_choice == "ML1: Logistic Regression":
            prob = logreg.predict_proba(input_selected)[0, 1]
            pred = (prob > 0.5).astype(int)
        elif model_choice == "ML2: LightGBM":
            prob = lgbm.predict(input_selected)[0]
            pred = int(prob > 0.5)
        elif model_choice == "ML3: CatBoost":
            prob = cat.predict_proba(input_selected)[0, 1]
            pred = cat.predict(input_selected)[0]
        elif model_choice == "ML4: Random Forest":
            prob = rf.predict_proba(input_selected)[0, 1]
            pred = rf.predict(input_selected)[0]
        elif model_choice == "ML5: Stacking":
            prob = stacking.predict_proba(input_selected)[0, 1]
            pred = stacking.predict(input_selected)[0]
        else:  
            prob = keras_model.predict(input_selected, verbose=0)[0, 0]
            pred = int(prob > 0.5)
        
        st.subheader("Результат предсказания")
        if pred == 1:
            st.error(f"💣 **Бомба заложена!** (вероятность {prob*100:.2f}%)")
        else:
            st.success(f"✅ **Бомба не заложена** (вероятность закладки {prob*100:.2f}%)")
        
        st.progress(float(prob))
        st.caption("Порог классификации = 0.5")

with st.sidebar.expander("📁 Загрузить CSV с примерами"):
    st.markdown("Файл должен содержать те же 28 признаков, что и в датасете.")
    uploaded_file = st.file_uploader("Выберите файл .csv", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"📄 Загружено строк: {len(df_batch)}")
        st.dataframe(df_batch.head())
        
        missing_cols = set(feature_names) - set(df_batch.columns)
        if missing_cols:
            st.error(f"❌ В файле отсутствуют колонки: {missing_cols}")
        else:
            batch_model = st.selectbox(
                "Модель для пакетного предсказания",
                ["ML1: Logistic Regression", "ML2: LightGBM", "ML3: CatBoost",
                 "ML4: Random Forest", "ML5: Stacking", "ML6: Neural Network (Keras/TF)"],
                key="batch_model"
            )
            
            if st.button("🚀 Выполнить предсказание для всех строк", key="batch_predict"):
                X_batch = df_batch[feature_names].values
                X_batch_scaled = scaler.transform(X_batch)
                X_batch_selected = selector.transform(X_batch_scaled)
                
                if batch_model == "ML1: Logistic Regression":
                    probs = logreg.predict_proba(X_batch_selected)[:, 1]
                    preds = (probs > 0.5).astype(int)
                elif batch_model == "ML2: LightGBM":
                    probs = lgbm.predict(X_batch_selected)
                    preds = (probs > 0.5).astype(int)
                elif batch_model == "ML3: CatBoost":
                    probs = cat.predict_proba(X_batch_selected)[:, 1]
                    preds = cat.predict(X_batch_selected)
                elif batch_model == "ML4: Random Forest":
                    probs = rf.predict_proba(X_batch_selected)[:, 1]
                    preds = rf.predict(X_batch_selected)
                elif batch_model == "ML5: Stacking":
                    probs = stacking.predict_proba(X_batch_selected)[:, 1]
                    preds = stacking.predict(X_batch_selected)
                else:  
                    probs = keras_model.predict(X_batch_selected, verbose=0).flatten()
                    preds = (probs > 0.5).astype(int)
                
                result_df = df_batch.copy()
                result_df['Probability_bomb'] = probs
                result_df['Predicted_bomb_planted'] = preds
                result_df['Prediction_text'] = result_df['Predicted_bomb_planted'].apply(
                    lambda x: "💣 Бомба заложена" if x == 1 else "✅ Бомба НЕ заложена"
                )
                
                st.subheader("📊 Результаты предсказаний")
                MAX_DISPLAY_ROWS = 100
                display_cols = ['Prediction_text', 'Probability_bomb'] + feature_names[:5]
                display_df = result_df[display_cols].copy()
                if len(display_df) > MAX_DISPLAY_ROWS:
                    st.warning(f"⚠️ Показаны только первые {MAX_DISPLAY_ROWS} строк из {len(display_df)}. Скачайте CSV, чтобы просмотреть все результаты.")
                    display_df = display_df.head(MAX_DISPLAY_ROWS)
                display_df['Probability_bomb'] = display_df['Probability_bomb'].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df)
                
                csv_output = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Скачать полный результат (CSV)",
                    data=csv_output,
                    file_name="predictions_results.csv",
                    mime="text/csv"
                )