import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

feature_name_mapping = {
    'time_of_day': 'Година доби',
    'temperature': 'Температура',
    'day_of_week': 'День тижня',
    'industrial_activity': 'Пром. активність',
    'solar_generation': 'Генерація СЕС',
    'noise_1_market_price': 'Шум_1_Ціна ринку',
    'noise_2_wind_speed': 'Шум_2_Шв. вітру',
    'noise_3_network_freq': 'Шум_3_Частота мережі',
    'noise_4_transformer_id': 'Шум_4_ID трансф.',
    'noise_5_random_noise': 'Шум_5_Випадковий'
}

def create_energy_dataset(n_samples=1000):
    """
    Створює симульований DataFrame для моніторингу енергосистеми.
    """
    print("Генерую симульовані дані енергосистеми...")
    data = {}

    data['time_of_day'] = np.random.randint(0, 24, n_samples)
    data['temperature'] = np.random.normal(15, 10, n_samples)
    data['day_of_week'] = np.random.randint(0, 7, n_samples)
    data['industrial_activity'] = np.random.uniform(0.5, 1.0, n_samples)
    data['solar_generation'] = np.sin(data['time_of_day'] / 24 * np.pi) * np.random.uniform(50, 100, n_samples)
    data['solar_generation'] = np.clip(data['solar_generation'], 0, None)

    data['noise_1_market_price'] = np.random.uniform(1, 5, n_samples)
    data['noise_2_wind_speed'] = np.random.uniform(0, 15, n_samples)
    data['noise_3_network_freq'] = np.random.normal(50.0, 0.05, n_samples)
    data['noise_4_transformer_id'] = np.random.randint(1, 50, n_samples)
    data['noise_5_random_noise'] = np.random.randn(n_samples)

    df = pd.DataFrame(data)

    load = 1000
    load += -500 * np.cos(df['time_of_day'] / 24 * 2 * np.pi) \
            + 200 * np.sin(df['time_of_day'] / 24 * 4 * np.pi)
    load += 5 * (df['temperature'] - 18)**2
    load -= df['day_of_week'].isin([5, 6]) * 300
    load += df['industrial_activity'] * 250
    load -= df['solar_generation']
    load += np.random.normal(0, 20, n_samples)
    df['system_load'] = load

    print("Дані для Вкладки 1 згенеровано.")
    return df

def create_tab(tab_control):
    """
    Створює вміст для першої вкладки (Відбір ознак)
    і прикріплює його до 'tab_control'.
    """

    tab1 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab1, text='Завдання 1: Відбір Ознак (Енерго)')

    print("Виконую обчислення для Вкладки 1 (Енерго)...")

    K_INFORMATIVE = 5
    df = create_energy_dataset(n_samples=1000)

    FEATURE_NAMES_ENG = list(df.columns.drop('system_load'))

    FEATURE_NAMES_UKR = [feature_name_mapping.get(name, name) for name in FEATURE_NAMES_ENG]
    TARGET_NAME = 'system_load'

    X = df[FEATURE_NAMES_ENG] 
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def test_model_on_features(X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)

    baseline_r2 = test_model_on_features(X_train_scaled, X_test_scaled, y_train, y_test)

    filter_selector = SelectKBest(score_func=f_regression, k=K_INFORMATIVE)
    X_train_filter = filter_selector.fit_transform(X_train_scaled, y_train)
    X_test_filter = filter_selector.transform(X_test_scaled)
    selected_indices_filter = filter_selector.get_support(indices=True)
    r2_filter = test_model_on_features(X_train_filter, X_test_filter, y_train, y_test)

    estimator_rfe = LinearRegression()
    wrapper_selector = RFE(estimator=estimator_rfe, n_features_to_select=K_INFORMATIVE, step=1)
    wrapper_selector.fit(X_train_scaled, y_train)
    X_train_wrapper = wrapper_selector.transform(X_train_scaled)
    X_test_wrapper = wrapper_selector.transform(X_test_scaled)
    selected_indices_wrapper = wrapper_selector.get_support(indices=True)
    r2_wrapper = test_model_on_features(X_train_wrapper, X_test_wrapper, y_train, y_test)

    embedded_model = RandomForestRegressor(n_estimators=100, random_state=42)
    embedded_model.fit(X_train_scaled, y_train)
    importances = embedded_model.feature_importances_
    selected_indices_embedded = np.argsort(importances)[::-1][:K_INFORMATIVE]
    selected_indices_embedded.sort()
    X_train_embedded = X_train_scaled[:, selected_indices_embedded]
    X_test_embedded = X_test_scaled[:, selected_indices_embedded]
    r2_embedded = test_model_on_features(X_train_embedded, X_test_embedded, y_train, y_test)

    print("Обчислення для Вкладки 1 завершено.")

    results_frame = ttk.LabelFrame(tab1, text="Результати (R-squared Score)", padding=(10, 10))
    results_frame.pack(pady=10, padx=10, fill="x")

    def add_result_row(frame, label_text, value_text, selected_indices_text):
        ttk.Label(frame, text=label_text, font=("Helvetica", 12, "bold")).grid(row=frame.grid_size()[1], column=0, sticky="w", padx=5, pady=2)
        ttk.Label(frame, text=f"{value_text:.4f}", font=("Helvetica", 12)).grid(row=frame.grid_size()[1]-1, column=1, sticky="w", padx=5)
        ttk.Label(frame, text=selected_indices_text, font=("Courier", 11)).grid(row=frame.grid_size()[1]-1, column=2, sticky="w", padx=20)

    def get_feature_names_ukr(indices):
        return [feature_name_mapping.get(FEATURE_NAMES_ENG[i], FEATURE_NAMES_ENG[i]) for i in indices]

    add_result_row(results_frame, "Еталон (всі 10 ознак):", baseline_r2, "-")
    add_result_row(results_frame, "1. Filter (SelectKBest):", r2_filter, f"Вибрані: {get_feature_names_ukr(selected_indices_filter)}")
    add_result_row(results_frame, "2. Wrapper (RFE):", r2_wrapper, f"Вибрані: {get_feature_names_ukr(selected_indices_wrapper)}")
    add_result_row(results_frame, "3. Embedded (RandomForest):", r2_embedded, f"Вибрані: {get_feature_names_ukr(selected_indices_embedded)}")

    legend_label = ttk.Label(results_frame,
                             text=f"Легенда: Інформативні [0-4] ({', '.join(FEATURE_NAMES_UKR[:K_INFORMATIVE])}),\n"
                                  f"Шумові [5-9] ({', '.join(FEATURE_NAMES_UKR[K_INFORMATIVE:])})",
                             font=("Helvetica", 9, "italic"), justify="left")
    legend_label.grid(row=results_frame.grid_size()[1], column=0, columnspan=3, sticky="w", padx=5, pady=10)

    charts_frame = ttk.Frame(tab1)
    charts_frame.pack(fill="both", expand=True, padx=10, pady=10)

    fig1 = Figure(figsize=(5, 4), dpi=100)
    ax1 = fig1.add_subplot(111)
    r2_scores = [baseline_r2, r2_filter, r2_wrapper, r2_embedded]
    labels = ['Еталон\n(10 ознак)', 'Фільтр\n(5 ознак)', 'Обгортка\n(5 ознак)', 'Вбудований\n(5 ознак)'] 
    colors = ['gray', 'blue', 'orange', 'green']
    ax1.bar(labels, r2_scores, color=colors)
    ax1.set_title('Порівняння якості моделей (R-squared)')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(min(r2_scores) * 0.95 if min(r2_scores) > 0 else min(r2_scores)*1.05,
                 max(r2_scores) * 1.05 if max(r2_scores)>0 else max(r2_scores)*0.95) 

    canvas1 = FigureCanvasTkAgg(fig1, master=charts_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    fig2 = Figure(figsize=(5.5, 4.5), dpi=100) 
    ax2 = fig2.add_subplot(111)

    bar_colors = ['green'] * K_INFORMATIVE + ['red'] * (len(FEATURE_NAMES_ENG) - K_INFORMATIVE)

    ax2.barh(FEATURE_NAMES_UKR, importances, color=bar_colors)
    ax2.set_title('Важливість ознак (Метод Embedded)')
    ax2.set_xlabel('Importance Score')
    ax2.invert_yaxis() 

    ax2.legend(['Інформативні', 'Шумові'],
               handles=[mpatches.Rectangle((0,0),1,1,color=c) for c in ['green', 'red']])
    fig2.tight_layout() 

    canvas2 = FigureCanvasTkAgg(fig2, master=charts_frame)
    canvas2.draw()

    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
