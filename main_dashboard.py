import tkinter as tk
from tkinter import ttk

import tab1_feature_selection
import tab2_selection_operators
import tab3_aco
import tab4_agents
import tab5_zipf_tfidf
import tab6_levenberg
import tab7_direct_communication
import tab8_neuro_fuzzy 

def main():
    root = tk.Tk()
    root.title("Дашборд: Моніторинг Завантаженості Енергосистеми (v2.0 Modular)")
    root.geometry("1100x850")

    style = ttk.Style()
    style.theme_use('clam')

    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill="both", padx=10, pady=10)

    try:
        tab1_feature_selection.create_tab(tab_control)
        tab2_selection_operators.create_tab(tab_control)
        tab3_aco.create_tab(tab_control)
        tab4_agents.create_tab(tab_control)
        tab5_zipf_tfidf.create_tab(tab_control)
        tab6_levenberg.create_tab(tab_control)
        tab7_direct_communication.create_tab(tab_control)
        tab8_neuro_fuzzy.create_tab(tab_control) 

    except Exception as e:
        error_frame = ttk.Frame(tab_control, padding=(10, 10))
        tab_control.add(error_frame, text='Помилка Завантаження')
        ttk.Label(error_frame, text=f"Не вдалося завантажити вкладку:\n{e}",
                  font=("Helvetica", 12), foreground="red").pack(padx=20, pady=20)
        print(f"Помилка при створенні вкладки: {e}")
        import traceback
        traceback.print_exc()

    print("Дашборд запущено (модульна структура).")
    root.mainloop()
    print("Програму закрито.")

if __name__ == "__main__":

    main()
