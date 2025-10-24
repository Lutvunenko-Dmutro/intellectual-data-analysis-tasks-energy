import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

def create_tree_data(n_samples=500):
    дані = {}
    дані['Година'] = np.random.randint(0, 24, n_samples)
    дані['Температура'] = np.random.normal(15, 10, n_samples)
    тип_дня_код = np.random.randint(0, 7, n_samples)
    дані['Тип_Дня'] = np.where(тип_дня_код < 5, 0, 1) # 0=Робочий, 1=Вихідний
    df = pd.DataFrame(дані)
    умови = [
        (df['Година'] >= 8) & (df['Година'] <= 11) & (df['Температура'] < 5) & (df['Тип_Дня'] == 0),
        (df['Година'] >= 18) & (df['Година'] <= 22) & (df['Температура'] < 10),
        (df['Година'] >= 12) & (df['Година'] <= 16) & (df['Температура'] > 28)
    ]
    варіанти = ['Так', 'Так', 'Так']
    df['Перевантаження'] = np.select(умови, варіанти, default='Ні')
    return df

дерево_модель = None
назви_ознак = ['Година', 'Температура', 'Тип_Дня']
назви_класів = ['Ні', 'Так']

def create_tab(tab_control):
    global дерево_модель, назви_ознак, назви_класів

    tab9 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab9, text='Завд. 1 (9): Дерева Рішень')

    головний_фрейм = ttk.Frame(tab9)
    головний_фрейм.pack(fill="both", expand=True)

    ЛІВА_ШИРИНА = 400
    лівий_фрейм = ttk.Frame(головний_фрейм, width=ЛІВА_ШИРИНА)
    лівий_фрейм.pack(side="left", fill="y", padx=(0, 10), expand=False)
    лівий_фрейм.pack_propagate(False)
    лівий_фрейм.grid_rowconfigure(0, weight=0)
    лівий_фрейм.grid_rowconfigure(1, weight=1)
    лівий_фрейм.grid_columnconfigure(0, weight=1)

    правий_фрейм = ttk.Frame(головний_фрейм)
    правий_фрейм.pack(side="right", fill="both", expand=True)
    правий_фрейм.grid_rowconfigure(0, weight=1)
    правий_фрейм.grid_rowconfigure(1, weight=0)
    правий_фрейм.grid_columnconfigure(0, weight=1)

    фрейм_налаштувань = ttk.LabelFrame(лівий_фрейм, text="Налаштування Моделі", padding=10)
    фрейм_налаштувань.grid(row=0, column=0, sticky='new', pady=(0, 10))

    ttk.Label(фрейм_налаштувань, text="Макс. глибина (max_depth):").grid(row=0, column=0, sticky='w', pady=2)
    змінна_макс_глибини = tk.IntVar(value=3)
    шкала_макс_глибини = ttk.Scale(фрейм_налаштувань, from_=1, to=10, orient='horizontal', variable=змінна_макс_глибини, length=150, command=lambda v: оновити_мітку_глибини_та_перебудувати())
    шкала_макс_глибини.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
    мітка_глибини = ttk.Label(фрейм_налаштувань, text=f"{змінна_макс_глибини.get()}")
    мітка_глибини.grid(row=0, column=2, padx=5)
    фрейм_налаштувань.grid_columnconfigure(1, weight=1)


    фрейм_теорії = ttk.LabelFrame(лівий_фрейм, text="Теорія: Дерева Рішень", padding=10)
    фрейм_теорії.grid(row=1, column=0, sticky='nsew')
    фрейм_теорії.grid_columnconfigure(0, weight=1)

    ШИРИНА_ПЕРЕНОСУ = ЛІВА_ШИРИНА - 40

    def створити_секцію_теорії_grid(батьківський, індекс_рядка, заголовок, зміст):
        рамка = ttk.LabelFrame(батьківський, text=заголовок, padding=7)
        рамка.grid(row=індекс_рядка, column=0, sticky='new', pady=5)
        рамка.grid_columnconfigure(0, weight=1)
        мітка = ttk.Label(рамка, text=зміст, wraplength=ШИРИНА_ПЕРЕНОСУ, justify="left", font=("Helvetica", 10))
        мітка.grid(row=0, column=0, sticky='ew')

    зміст_теорії_1 = ( "Дерево рішень - це модель машинного навчання, що використовує деревоподібну структуру для класифікації або регресії.\n\n- Вузли (Nodes): Перевіряють умову щодо однієї з ознак (напр., 'Температура < 10?').\n- Гілки (Branches): Відповідають результатам перевірки ('Так' / 'Ні').\n- Листи (Leaves): Кінцеві вузли, що містять прогноз (напр., клас 'Перевантаження: Так').")
    створити_секцію_теорії_grid(фрейм_теорії, 0, "Принцип роботи", зміст_теорії_1)
    зміст_теорії_2 = ("Дерево будується рекурсивно зверху вниз ('жадібним' способом). На кожному кроці алгоритм шукає найкраще розділення (ознаку та поріг), яке максимально зменшує невизначеність (impurity) у дочірніх вузлах.\n\nКритерії невизначеності для класифікації:\n- Індекс Джині (Gini Impurity): Вимірює ймовірність неправильної класифікації. Прагне до 0 (чистий вузол).\n- Ентропія (Entropy): Міра хаосу або інформаційної невизначеності. Також прагне до 0 (використовується в ID3, C4.5).")
    створити_секцію_теорії_grid(фрейм_теорії, 1, "Побудова та Критерії", зміст_теорії_2)
    зміст_теорії_3 = ("Переваги:\n- Інтерпретованість: Правила легко зрозуміти.\n- Не потребують масштабування даних.\n- Обробка різних типів даних.\n\nНедоліки:\n- Схильність до перенавчання (особливо глибокі дерева).\n- Нестабільність: Малі зміни даних можуть сильно змінити дерево.\n- Проблеми з деякими залежностями (напр., лінійними).")
    створити_секцію_теорії_grid(фрейм_теорії, 2, "Переваги та Недоліки", зміст_теорії_3)


    фрейм_візуалізації_дерева = ttk.LabelFrame(правий_фрейм, text="Побудоване Дерево Рішень", padding=10)
    фрейм_візуалізації_дерева.grid(row=0, column=0, sticky='nsew')

    фігура = Figure(figsize=(10, 6), dpi=100)
    вісь = фігура.add_subplot(111)
    полотно = FigureCanvasTkAgg(фігура, master=фрейм_візуалізації_дерева)
    віджет_полотна = полотно.get_tk_widget()
    віджет_полотна.pack(fill="both", expand=True)

    фрейм_таблиці_аналізу = ttk.LabelFrame(правий_фрейм, text="Аналіз та Результати (на Тестових Даних)", padding=10)
    фрейм_таблиці_аналізу.grid(row=1, column=0, sticky='sew', pady=(10, 0))

    колонки_аналізу = ('метрика', 'клас_ні', 'клас_так', 'загалом')
    дерево_аналізу = ttk.Treeview(фрейм_таблиці_аналізу, columns=колонки_аналізу, show='headings', height=5)

    дерево_аналізу.column('метрика', width=100, anchor='w'); дерево_аналізу.heading('метрика', text='Метрика')
    дерево_аналізу.column('клас_ні', width=120, anchor='center'); дерево_аналізу.heading('клас_ні', text='Клас "Ні"')
    дерево_аналізу.column('клас_так', width=120, anchor='center'); дерево_аналізу.heading('клас_так', text='Клас "Так"')
    дерево_аналізу.column('загалом', width=150, anchor='center'); дерево_аналізу.heading('загалом', text='Загалом / Accuracy')

    дерево_аналізу.tag_configure('oddrow', background='#f0f0f0')
    дерево_аналізу.tag_configure('evenrow', background='white')
    дерево_аналізу.pack(fill="both", expand=True)

    def побудувати_та_проаналізувати_дерево():
        global дерево_модель, назви_класів, назви_ознак

        print("Побудова дерева рішень (Вкладка 9)...")
        try: df = create_tree_data()
        except Exception as e: print(f"Помилка генерації даних: {e}"); return

        X = df[назви_ознак]; y_raw = df['Перевантаження']
        le = LabelEncoder(); y = le.fit_transform(y_raw)
        назви_класів[:] = list(le.classes_)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        try: глибина = змінна_макс_глибини.get()
        except tk.TclError: глибина = 3; змінна_макс_глибини.set(3)

        дерево_модель = DecisionTreeClassifier(max_depth=глибина, criterion='gini', random_state=42, min_samples_leaf=5)
        дерево_модель.fit(X_train, y_train)

        вісь.clear()
        try:
            plot_tree(дерево_модель, feature_names=назви_ознак, class_names=назви_класів,
                      filled=True, rounded=True, impurity=True, proportion=False, fontsize=8, ax=вісь)
            фігура.tight_layout(pad=1.5)
            полотно.draw()
        except Exception as e:
            вісь.text(0.5, 0.5, f"Помилка візуалізації:\n{e}", ha='center', va='center', color='red'); полотно.draw()
            print(f"Помилка візуалізації: {e}")

        дерево_аналізу.delete(*дерево_аналізу.get_children())
        try:
            y_pred = дерево_модель.predict(X_test)
            точність = accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)

            дерево_аналізу.insert('', 'end', values=('Precision (Точність)', f"{precision[0]:.3f}", f"{precision[1]:.3f}", ''), tags=('evenrow',))
            дерево_аналізу.insert('', 'end', values=('Recall (Повнота)', f"{recall[0]:.3f}", f"{recall[1]:.3f}", ''), tags=('oddrow',))
            дерево_аналізу.insert('', 'end', values=('F1-Score', f"{fscore[0]:.3f}", f"{fscore[1]:.3f}", ''), tags=('evenrow',))
            дерево_аналізу.insert('', 'end', values=('Support (К-ть)', f"{support[0] if support is not None else 'N/A'}", f"{support[1] if support is not None else 'N/A'}", ''), tags=('oddrow',))
            дерево_аналізу.insert('', 'end', values=('Accuracy (Загальна)', '', '', f"{точність:.3f}"), tags=('evenrow',))

            try:
                правила_дерева = export_text(дерево_модель, feature_names=назви_ознак, decimals=1)
                print("\n--- Текстове представлення правил ---")
                print(правила_дерева); print("-------------------------------------\n")
            except Exception as rule_err: print(f"--- Не вдалося отримати текст правил: {rule_err} ---\n")

        except Exception as e:
             print(f"Помилка оцінки: {e}")
             дерево_аналізу.insert('', 'end', values=('Помилка оцінки', str(e), '', ''), tags=('evenrow',))

        print("Побудова та аналіз дерева завершено.")

    def оновити_мітку_глибини_та_перебудувати(event=None):
         мітка_глибини.config(text=f"{змінна_макс_глибини.get():.0f}")
         побудувати_та_проаналізувати_дерево()

    шкала_макс_глибини.config(command=оновити_мітку_глибини_та_перебудувати)

    побудувати_та_проаналізувати_дерево()