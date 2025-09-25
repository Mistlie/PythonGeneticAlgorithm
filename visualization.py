import sys
from collections import defaultdict
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


"""Вывод расписания с ограничениями"""
def print_schedule(schedule: Dict, config: Dict):
    if not schedule or not schedule.get('schedule'):
        print("\n[Внимание] Не удалось создать расписание!")
        return

    student_groups = config['STUDENT_GROUPS']
    time_slots = [f"{d} - {t}" for d in config['DAYS'] for t in config['TIMES']]
    days_order = config['DAYS']
    course_constraints = config['COURSE_CONSTRAINTS']

    print("\n" + "=" * 80)
    print("ОПТИМАЛЬНОЕ РАСПИСАНИЕ ЗАНЯТИЙ")
    print("=" * 80)

    # Группировка по группам и дням
    schedule_by_group = defaultdict(lambda: defaultdict(list))
    for lesson in schedule['schedule']:
        day = lesson['time_slot'].split(' - ')[0]
        schedule_by_group[lesson['group']][day].append(lesson)

    time_slot_order = {slot: i for i, slot in enumerate(time_slots)}
    days_order = config['DAYS']

    for group in sorted(schedule_by_group.keys()):
        print(f"\nГРУППА: {group}")
        print("-" * 80)

        for day in days_order:
            if day in schedule_by_group[group]:
                print(f"\n{day}:")
                day_lessons = schedule_by_group[group][day]
                day_lessons.sort(key=lambda x: time_slot_order.get(x['time_slot'], sys.maxsize))

                for i, lesson in enumerate(day_lessons, 1):
                    print(f"  {i}. {lesson['time_slot']:20} | {lesson['course']:15} | "
                          f"Преподаватель: {lesson['teacher']:12} | Аудитория: {lesson['classroom']}")

    # Статистика с ограничениями
    print("\n" + "=" * 80)
    print("СТАТИСТИКА РАСПИСАНИЯ")
    print("-" * 80)

    print(f"Общее количество занятий: {len(schedule['schedule'])}")
    print(f"Качество расписания: {schedule.get('fitness', 0):.4f}")

    # Статистика по предметам с ограничениями
    course_stats = defaultdict(int)
    for lesson in schedule['schedule']:
        course_stats[lesson['course']] += 1

    print("\nРаспределение занятий по предметам (минимум):")
    for course in sorted(course_stats.keys()):
        actual = course_stats[course]
        min_hours = course_constraints[course]['min']

        status = f"✅ В НОРМЕ (+{actual - min_hours} сверх минимума)" if actual >= min_hours else f"❌ НЕДОСТАТОК (нужно {min_hours}+)"
        print(f"  {course}: {actual} часов (минимум {min_hours}) {status}")

    # Проверка ограничения на 6 пар
    group_day_stats = defaultdict(lambda: defaultdict(int))
    for lesson in schedule['schedule']:
        day = lesson['time_slot'].split(' - ')[0]
        group_day_stats[lesson['group']][day] += 1

    violation_count = 0
    print("\nПроверка ограничения 'не более 6 пар в день' для групп:")
    for group in sorted(student_groups):
        for day in days_order:
            count = group_day_stats[group].get(day, 0)
            if count > 6:
                print(f"  ❌ {group} в {day}: {count} пар (превышение!)")
                violation_count += 1
            elif count == 6:
                print(f"  ⚠️  {group} в {day}: {count} пар (максимум)")

    if violation_count == 0:
        print("  ✅ Все группы соблюдают ограничение на 6 пар в день!")
    else:
        print(f"  ❌ Найдено {violation_count} нарушений ограничения!")


"""Визуализация процесса и результата"""
class ScheduleVisualizer:

    def __init__(self, config: Dict):
        self.fitness_history: List[float] = []
        self.penalty_history: List[float] = []
        self.constraint_satisfaction_history: List[float] = []
        self.generation_history: List[int] = []
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.days = config['DAYS']
        self.times = config['TIMES']

    """Обновление истории обучения"""
    def update_training_history(self, generation, fitness, penalties, constraint_satisfaction, train_loss=None,
                                val_loss=None):
        self.generation_history.append(generation)
        self.fitness_history.append(fitness)
        self.penalty_history.append(penalties)
        self.constraint_satisfaction_history.append(constraint_satisfaction)

        if train_loss:
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

    """Визуализация прогресса оптимизации"""
    def plot_optimization_progress(self):
        if not self.generation_history:
            print("Нет данных для построения графиков прогресса.")
            return

        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 3, figure=fig)

        # График приспособленности
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_fitness_progress(ax1)

        # рафик потерь нейросети
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_loss_progress(ax2)

        # График удовлетворения ограничений
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_constraint_satisfaction(ax3)

        plt.tight_layout()
        plt.show()

    """График прогресса приспособленности"""
    def _plot_fitness_progress(self, ax):
        ax.plot(self.generation_history, self.fitness_history, 'b-', linewidth=2, label='Средняя приспособленность')
        ax.plot(self.generation_history, self.penalty_history, 'r--', alpha=0.7, label='Средние штрафы')
        ax.set_xlabel('Поколение')
        ax.set_ylabel('Значение')
        ax.set_title('Прогресс оптимизации')
        ax.legend()
        ax.grid(True, alpha=0.3)

    """График потерь нейросети"""
    def _plot_loss_progress(self, ax):
        if self.train_loss_history:
            epochs = np.arange(1, len(self.train_loss_history) + 1) * 20  # Обучение каждые 20 поколений
            ax.plot(epochs, self.train_loss_history, 'g-', label='Train Loss')
            if self.val_loss_history:
                ax.plot(epochs, self.val_loss_history, 'orange', label='Val Loss')
            ax.set_xlabel('Поколение обучения')
            ax.set_ylabel('Потери')
            ax.set_title('Обучение нейросети (каждые 20 ген.)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title('Обучение NN (Нет данных)')
            ax.axis('off')

    """График удовлетворения ограничений"""
    def _plot_constraint_satisfaction(self, ax):
        ax.plot(self.generation_history, self.constraint_satisfaction_history,
                'purple', linewidth=2, label='Средн. удовл. огранич.')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Цель (1.0)')
        ax.set_xlabel('Поколение')
        ax.set_ylabel('Удовлетворение')
        ax.set_title('Удовлетворение мин. ограничений')
        ax.legend()
        ax.grid(True, alpha=0.3)

    """Визуализация финального расписания"""
    def plot_final_schedule_analysis(self, schedule: Dict, courses: List[str], teachers: List[str]):
        if not schedule or not schedule.get('schedule'):
            print("\nНет данных расписания для визуализации анализа")
            return

        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 3, figure=fig)

        # Распределение занятий по предметам
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_course_distribution(ax1, schedule, courses)

        # Нагрузка преподавателей
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_teacher_workload(ax2, schedule, teachers)

        # Распределение по дням недели
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_daily_distribution(ax3, schedule)

        plt.tight_layout()
        plt.show()

    """Распределение занятий по предметам"""
    def _plot_course_distribution(self, ax, schedule, courses):
        course_hours = defaultdict(int)
        for lesson in schedule['schedule']:
            course_hours[lesson['course']] += 1

        courses_sorted = sorted(course_hours.items(), key=lambda x: x[1], reverse=True)
        courses_names = [item[0] for item in courses_sorted]
        hours = [item[1] for item in courses_sorted]

        bars = ax.bar(courses_names, hours, color='skyblue', alpha=0.7)
        ax.set_title('Распределение часов по предметам')
        ax.set_ylabel('Количество часов')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

    """Нагрузка преподавателей"""
    def _plot_teacher_workload(self, ax, schedule, teachers):
        teacher_load = defaultdict(int)
        for lesson in schedule['schedule']:
            teacher_load[lesson['teacher']] += 1

        teachers_sorted = sorted(teacher_load.items(), key=lambda x: x[1], reverse=True)
        teachers_names = [item[0] for item in teachers_sorted]
        load = [item[1] for item in teachers_sorted]

        colors = plt.cm.Set3(np.linspace(0, 1, len(teachers)))
        bars = ax.bar(teachers_names, load, color=colors, alpha=0.7)
        ax.set_title('Нагрузка преподавателей')
        ax.set_ylabel('Количество занятий')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

    """Распределение занятий по дням недели (круговая диаграмма)"""
    def _plot_daily_distribution(self, ax, schedule):
        day_count = {day: 0 for day in self.days}
        for lesson in schedule['schedule']:
            day = lesson['time_slot'].split(' - ')[0]
            if day in day_count:
                day_count[day] += 1

        day_values = [day_count[day] for day in self.days if day_count[day] > 0]
        day_labels = [day for day in self.days if day_count[day] > 0]

        if not day_values:
            ax.set_title('Нет данных о днях')
            return

        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        ax.pie(day_values, labels=day_labels, colors=colors[:len(day_values)],
               autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
        ax.set_title('Распределение по дням недели')

    """Визуализация расписания для каждой группы в виде таблицы в отдельном окне"""
    def plot_group_schedules(self, schedule: Dict, groups: List[str]):
        if not schedule or not schedule.get('schedule'):
            print("\nНет данных расписания для визуализации по группам.")
            return

        print("\nПостроение таблиц расписания для каждой группы...")

        # Предварительная обработка данных для удобства
        schedules_by_group = defaultdict(lambda: defaultdict(dict))
        for lesson in schedule['schedule']:
            group = lesson['group']
            day = lesson['time_slot'].split(' - ')[0]
            time = lesson['time_slot'].split(' - ')[1]
            lesson_info = f"{lesson['course']}\n{lesson['teacher']}\nАуд. {lesson['classroom']}"
            schedules_by_group[group][day][time] = lesson_info

        # Создание отдельного окна для каждой группы
        for group in sorted(groups):
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('off')
            ax.set_title(f"Расписание для группы: {group}", fontsize=16, fontweight='bold', pad=20)

            table_data = []
            for day in self.days:
                row = []
                for time in self.times:
                    cell_text = schedules_by_group[group][day].get(time, "")
                    row.append(cell_text)
                table_data.append(row)

            table = plt.table(cellText=table_data,
                              rowLabels=self.days,
                              colLabels=self.times,
                              loc='center',
                              cellLoc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)

            for (i, j), cell in table.get_celld().items():
                if i == 0 or j == -1:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#40466e')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('grey')

            plt.tight_layout()
            plt.show()