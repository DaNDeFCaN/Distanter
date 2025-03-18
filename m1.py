import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, shapiro
import os


class DataProc(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Distanter")
        self.geometry("600x400")
        self.houses_file = ""
        self.kindergartens_file = ""
        self.R = 0.0
        self.create_widgets()
        self.norm_reg = 0.055
        try:
            self.iconbitmap('icon.ico')  # Для Windows
        except:
            img = tk.PhotoImage(file='icon.png')  # Резерв для Linux/Mac
            self.tk.call('wm', 'iconphoto', self._w, img)

    def create_widgets(self):
        file_frame = ttk.LabelFrame(self, text="Выбор файлов")
        file_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(file_frame, text="Файл домов (CSV):").grid(row=0, column=0, padx=5)
        self.houses_entry = ttk.Entry(file_frame, width=50)
        self.houses_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Обзор",
                   command=lambda: self.load_file(self.houses_entry, "houses")).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Файл соцучреждений (CSV):").grid(row=1, column=0, padx=5)
        self.kindergartens_entry = ttk.Entry(file_frame, width=50)
        self.kindergartens_entry.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Обзор",
                   command=lambda: self.load_file(self.kindergartens_entry, "kindergartens")).grid(row=1, column=2,
                                                                                                   padx=5)

        param_frame = ttk.LabelFrame(self, text="Параметры расчета")
        param_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(param_frame, text="Режим работы:").grid(row=0, column=0)
        self.mode_var = tk.StringVar()
        self.mode_combobox = ttk.Combobox(param_frame, textvariable=self.mode_var,
                                          values=["Базовый расчет", "Оптимизация радиуса"])
        self.mode_combobox.grid(row=0, column=1, pady=5)
        self.mode_combobox.current(0)
        ttk.Label(param_frame, text="Тип учреждения:").grid(row=1, column=0)
        self.institution_var = tk.StringVar()
        self.institution_combobox = ttk.Combobox(param_frame,
                                                 textvariable=self.institution_var,
                                                 values=["Детсады", "Школы"])
        self.institution_combobox.grid(row=1, column=1, pady=5)
        self.institution_combobox.current(0)  # Установка значения по умолчанию
        self.institution_combobox.bind("<<ComboboxSelected>>", self.instit)

        ttk.Label(param_frame, text="Радиус R (нормативно 300/500 м):").grid(row=2, column=0)
        self.R_entry = ttk.Entry(param_frame)
        self.R_entry.grid(row=2, column=1)

        ttk.Button(self, text="Начать обработку", command=self.start_processing).pack(pady=10)

        log_frame = ttk.LabelFrame(self, text="Ход выполнения")
        log_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.log = scrolledtext.ScrolledText(log_frame, width=100, height=20)
        self.log.pack(fill="both", expand=True)

    def load_file(self, entry, file_type):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)
            if file_type == "houses":
                self.houses_file = filename
            else:
                self.kindergartens_file = filename

    def log_message(self, message):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def validate_inputs(self):
        if not os.path.exists(self.houses_file):
            raise ValueError("Файл домов не выбран или не существует")
        if not os.path.exists(self.kindergartens_file):
            raise ValueError("Файл детсадов не выбран или не существует")
        try:
            self.R = float(self.R_entry.get())
        except:
            raise ValueError("Некорректное значение радиуса R")

    def instit(self, event):
        if self.institution_var.get() == "Детсады":
            self.norm_reg = 0.055
        else:
            self.norm_reg = 0.112

    def start_processing(self):
        try:
            self.validate_inputs()

            if "Базовый" in self.mode_var.get():
                self.run_basic_calculation()
            else:
                self.run_optimization()

            messagebox.showinfo("Успешно", "Обработка завершена!")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def run_basic_calculation(self):
        self.log_message("Запуск базового расчета...")

        houses = pd.read_csv(self.houses_file, dtype={'id': str})
        kgs = pd.read_csv(self.kindergartens_file, dtype={'id': str})
        R = float(self.R_entry.get())

        all_kg_ids = kgs['id'].astype(str).tolist()
        for kg_id in all_kg_ids:
            houses[f'propdost{kg_id}'] = 0.0

        kg_coords = {row['id']: (row['x'], row['y']) for _, row in kgs.iterrows()}
        temp_kgs_list = [[] for _ in range(len(houses))]

        for idx, house in houses.iterrows():
            x_house = house['x']
            y_house = house['y']
            nearby_kgs = []

            for kg_id, (x_kg, y_kg) in kg_coords.items():
                distance = math.sqrt((x_house - x_kg) ** 2 + (y_house - y_kg) ** 2)
                if distance <= R:
                    nearby_kgs.append(kg_id)

            houses.at[idx, 'idkindgartn'] = ', '.join(nearby_kgs)
            temp_kgs_list[idx] = nearby_kgs

            if nearby_kgs:
                pop_per_kg = house['population'] / len(nearby_kgs)
                for kg_id in nearby_kgs:
                    houses.at[idx, f'propdost{kg_id}'] = pop_per_kg

        kgs['sumdost'] = kgs['id'].apply(lambda kg_id: houses[f'propdost{kg_id}'].sum())
        kgs['depand'] = kgs['sumdost'] / kgs['capacity']
        kgs['1depand'] = kgs['depand'].apply(lambda x: 1 / x if x != 0 else 0)

        for kg_id in all_kg_ids:
            houses[f'propobes{kg_id}'] = 0.0

        kg_1depand = kgs.set_index('id')['1depand'].to_dict()

        for idx in range(len(houses)):
            nearby_kgs = temp_kgs_list[idx]
            if not nearby_kgs:
                continue

            total_1depand = sum(kg_1depand[kg_id] for kg_id in nearby_kgs)
            if total_1depand == 0:
                continue

            for kg_id in nearby_kgs:
                value = (houses.at[idx, 'population'] * kg_1depand[kg_id]) / total_1depand
                houses.at[idx, f'propobes{kg_id}'] = value

        kgs['sumobes'] = kgs['id'].apply(lambda kg_id: houses[f'propobes{kg_id}'].sum())
        kgs['srednagr'] = (kgs['sumdost'] + kgs['sumobes']) / 2
        kgs['obes'] = kgs['capacity'] / kgs['srednagr']

        houses['obespech'] = 0.0
        for idx in range(len(houses)):
            nearby_kgs = temp_kgs_list[idx]
            if not nearby_kgs:
                continue

            relevant_obes = kgs.loc[kgs['id'].isin(nearby_kgs), 'obes']
            houses.at[idx, 'obespech'] = relevant_obes.fillna(0).mean()

        houses['obespech'] = houses['obespech'].fillna(0)

        column_order = ['id', 'population', 'x', 'y', 'idkindgartn'] + \
                       [col for col in houses.columns if col.startswith('propdost')] + \
                       [col for col in houses.columns if col.startswith('propobes')] + \
                       ['obespech']
        houses = houses[column_order]

        total_pop = houses['population'].sum()
        weighted_avg = (houses['obespech'] * houses['population']).sum() / total_pop
        obespech_values = houses['obespech']
        skewness = skew(obespech_values)
        kurtosis_value = kurtosis(obespech_values)

        plt.figure(num=1, figsize=(12, 7), dpi=100)
        plt.clf()
        unique_values = np.sort(obespech_values.unique())
        n_unique = len(unique_values)

        if n_unique <= 8:
            if n_unique <= 4:
                bin_width = 0.0015
            else:
                bin_width = 0.004
            bins = []
            for val in unique_values:
                bins.extend([val - bin_width / 2, val + bin_width / 2])
            bins = np.unique(bins)
            bins = np.append(bins, bins[-1] + bin_width)
        else:
            n_data = len(obespech_values)
            k = 1 + int(np.ceil(np.log2(n_data)))
            bins = k

        n, bins, patches = plt.hist(
            obespech_values,
            weights=houses['population'],
            bins=bins,
            edgecolor='black',
            color='black',
            rwidth=0.95
        )

        for i in range(len(n)):
            x_center = (bins[i] + bins[i + 1]) / 2
            y_height = n[i]
            if y_height != 0:
                plt.text(
                    x_center,
                    y_height + 50,
                    f'{x_center:.5f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black'
                )

        r = self.R
        if r.is_integer():
            r_str = f"{int(r)}"
        else:
            r_str = f"{r:.1f}".replace('.', ',')

        plt.axvline(weighted_avg, color='red', linewidth=4, label=f'Среднее: {weighted_avg:.4f}')
        plt.axvline(self.norm_reg, color='green', linewidth=4, label=f'Нормативное: {self.norm_reg}')
        settlement = kgs['capacity'].sum() / houses['population'].sum()
        plt.axvline(settlement, color='blue', linewidth=4, label=f'Расчетное: {settlement:.4f}')

        text_str = f"Эксцесс: {kurtosis_value:.2f}\nАсимметрия: {skewness:.2f}"
        plt.text(0.9, 0.95, text_str, transform=plt.gca().transAxes,
                 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.xlabel('Обеспеченность', fontsize=12)
        plt.ylabel('Население', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'provision_histogram_{r_str}.png', dpi=300, bbox_inches='tight')
        plt.close(1)

        houses.to_csv(f"result_houses_{r_str}.csv", index=False)
        kgs.to_csv(f"result_kindergartens_{r_str}.csv", index=False)
        self.log_message(f'Конец базового расчета.')

    def run_optimization(self):
        self.log_message("Запуск оптимизации радиуса...")
        try:
            initial_R = float(self.R_entry.get())
            best_R = initial_R
            best_p = self.optim(initial_R)
            step = 0.5
            direction = step
            max_iterations = 400
            iteration = 0

            p_plus = self.optim(initial_R + step)
            p_minus = self.optim(initial_R - step)

            if best_p == 0.0 or (p_plus == 0.0 and p_minus == 0.0):
                direction = step
            elif p_plus >= p_minus:
                direction = step
            else:
                direction = -step

            while iteration < max_iterations:
                iteration += 1
                current_R = best_R + direction

                if current_R > 500 or current_R < 0:
                    break

                current_p = self.optim(current_R)

                if current_p > best_p and current_p != 0.0:
                    best_p = current_p
                    best_R = current_R
                else:
                    break

                current_R += step

            search_start = max(0.0, best_R - 500)
            search_end = min(1500.0, best_R + 500)
            current_search_R = search_start

            self.log_message(f"Полный поиск в диапазоне {search_start:.1f}-{search_end:.1f}")

            while current_search_R <= search_end:
                current_p = self.optim(current_search_R)
                if current_p > best_p and current_p != 0.0:
                    best_p = current_p
                    best_R = current_search_R
                    self.log_message(f"Найден лучший R: {current_search_R:.1f}, p={current_p:.8f}")
                current_search_R += step

            self.log_message(f"Оптимальный радиус: {best_R:.1f}")
            self.save_final_results(best_R)

        except Exception as e:
            self.log_message(f"Ошибка оптимизации: {str(e)}")
            raise

    def optim(self, R):
        try:
            houses = pd.read_csv(self.houses_file, dtype={'id': str})
            kgs = pd.read_csv(self.kindergartens_file, dtype={'id': str})

            all_kg_ids = kgs['id'].astype(str).tolist()
            for kg_id in all_kg_ids:
                houses[f'propdost{kg_id}'] = 0.0

            kg_coords = {row['id']: (row['x'], row['y']) for _, row in kgs.iterrows()}
            temp_kgs_list = [[] for _ in range(len(houses))]

            for idx, house in houses.iterrows():
                x_house = house['x']
                y_house = house['y']
                nearby_kgs = []

                for kg_id, (x_kg, y_kg) in kg_coords.items():
                    distance = math.sqrt((x_house - x_kg) ** 2 + (y_house - y_kg) ** 2)
                    if distance <= R:
                        nearby_kgs.append(kg_id)

                houses.at[idx, 'idkindgartn'] = ', '.join(nearby_kgs)
                temp_kgs_list[idx] = nearby_kgs

                if nearby_kgs:
                    pop_per_kg = house['population'] / len(nearby_kgs)
                    for kg_id in nearby_kgs:
                        houses.at[idx, f'propdost{kg_id}'] = pop_per_kg

            kgs['sumdost'] = kgs['id'].apply(lambda kg_id: houses[f'propdost{kg_id}'].sum())
            kgs['depand'] = kgs['sumdost'] / kgs['capacity']
            kgs['1depand'] = kgs['depand'].apply(lambda x: 1 / x if x != 0 else 0)

            for kg_id in all_kg_ids:
                houses[f'propobes{kg_id}'] = 0.0

            kg_1depand = kgs.set_index('id')['1depand'].to_dict()

            for idx in range(len(houses)):
                nearby_kgs = temp_kgs_list[idx]
                if not nearby_kgs:
                    continue

                total_1depand = sum(kg_1depand[kg_id] for kg_id in nearby_kgs)
                if total_1depand == 0:
                    continue

                for kg_id in nearby_kgs:
                    value = (houses.at[idx, 'population'] * kg_1depand[kg_id]) / total_1depand
                    houses.at[idx, f'propobes{kg_id}'] = value

            kgs['sumobes'] = kgs['id'].apply(lambda kg_id: houses[f'propobes{kg_id}'].sum())
            kgs['srednagr'] = (kgs['sumdost'] + kgs['sumobes']) / 2
            kgs['obes'] = kgs['capacity'] / kgs['srednagr']

            houses['obespech'] = 0.0
            for idx in range(len(houses)):
                nearby_kgs = temp_kgs_list[idx]
                if not nearby_kgs:
                    continue

                relevant_obes = kgs.loc[kgs['id'].isin(nearby_kgs), 'obes']
                houses.at[idx, 'obespech'] = relevant_obes.fillna(0).mean()

            houses['obespech'] = houses['obespech'].fillna(0)
            obespech_values = houses['obespech']

            if len(obespech_values.unique()) < 3:
                return 0.0

            _, shapiro_p_value = shapiro(obespech_values)
            return shapiro_p_value

        except Exception as e:
            self.log_message(f"Ошибка в optim: {str(e)}")
            return 0.0

    def save_final_results(self, best_R):
        self.log_message("Сохранение финальных результатов...")

        r = self.R
        if r.is_integer():
            r_str = f"{int(r)}"
        else:
            r_str = f"{r:.1f}".replace('.', ',')

        try:
            original_mode = self.mode_var.get()
            original_R = self.R_entry.get()

            self.mode_var.set("Базовый расчет")
            self.R_entry.delete(0, tk.END)
            self.R_entry.insert(0, str(best_R))
            self.run_basic_calculation()

            houses = pd.read_csv(f"result_houses_{r_str}.csv", dtype={'id': str})
            kgs = pd.read_csv(f"result_kindergartens_{r_str}.csv", dtype={'id': str})

            if 'obespech' not in houses.columns:
                raise ValueError("Столбец 'obespech' отсутствует в результатах")

            # Генерация графика
            plt.figure(num=3, figsize=(12, 7), dpi=100)
            plt.clf()


            obespech_values = houses['obespech']
            total_pop = houses['population'].sum()
            skewness = skew(obespech_values)
            kurtosis_value = kurtosis(obespech_values)
            weighted_avg = (houses['obespech'] * houses['population']).sum() / total_pop
            settlement = kgs['capacity'].sum() / total_pop

            # Построение гистограммы
            n_unique = len(obespech_values.unique())
            bins = self.calculate_bins(obespech_values, n_unique)

            n, bins, patches = plt.hist(
                obespech_values,
                weights=houses['population'],
                bins=bins,
                edgecolor='black',
                color='black',
                rwidth=0.95
            )

            plt.axvline(weighted_avg, color='red', lw=4,
                         label=f'Среднее: {weighted_avg:.4f}')
            plt.axvline(self.norm_reg, color='green', lw=4,
                        label=f'Нормативное: {self.norm_reg}')
            plt.axvline(settlement, color='blue', lw=4,
                        label=f'Расчетное: {settlement:.4f}')

            for i in range(len(n)):
                x_center = (bins[i] + bins[i + 1]) / 2
                y_height = n[i]

                if y_height != 0:
                    plt.text(
                        float(x_center),
                        y_height + 50,
                        f'{x_center:.5f}',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black'
                    )

            text_str = f"Эксцесс: {kurtosis_value:.2f}\nАсимметрия: {skewness:.2f}"
            plt.text(0.9
                     , 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                     # Положение по x и y для надписи - можно и нужно порой подтачивать руками
                     ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            plt.xlabel('Обеспеченность', fontsize=12)
            plt.ylabel('Население', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10, loc='upper left')

            plt.savefig(
                'provision_histogram.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                metadata={'Description': f'Optimal R={best_R}'}
            )
            plt.close(3)

            # Восстановление исходных параметров
            self.mode_var.set(original_mode)
            self.R_entry.delete(0, tk.END)
            self.R_entry.insert(0, original_R)

            self.log_message("Финальные результаты успешно сохранены")

        except Exception as e:
            self.log_message(f"Критическая ошибка: {str(e)}")
            raise

    def calculate_bins(self, data, n_unique):
        if n_unique <= 8:
            # Режим ручного расчета бинов для малого числа уникальных значений
            if n_unique <= 4:
                bin_width = 0.002
            else:
                bin_width = 0.004
            bins = []
            for val in np.sort(data.unique()):
                bins.extend([val - bin_width / 2, val + bin_width / 2])
            bins = np.append(np.unique(bins), bins[-1] + bin_width)
        else:
            # Явная реализация правила Стерджеса
            n_data = len(data)
            k = 1 + int(np.ceil(np.log2(n_data)))
            min_val = data.min()
            max_val = data.max()
            bin_edges = np.linspace(min_val, max_val, k + 1)
            bins = np.hstack([
                bin_edges[:-1] - 1e-9,
                bin_edges[-1:]
            ])

        return bins

    def add_plot_lines(self, weighted_avg, settlement):
        plt.axvline(weighted_avg, color='red', lw=4, label=f'Среднее: {weighted_avg:.4f}')
        plt.axvline(0.112, color='green', lw=4, label='Нормативное: 0,112')
        plt.axvline(settlement, color='blue', lw=4, label=f'Расчетное: {settlement:.4f}')
        plt.legend()

    def save_plot(self, best_R):
        plt.savefig(
            'provision_histogram.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            metadata={
                'Title': 'График обеспеченности',
                'Author': 'DaNCaN',
                'Description': f'Сгенерировано для R={best_R}'
            }
        )
        plt.close(2)

if __name__ == "__main__":
    app = DataProc()
    app.mainloop()
