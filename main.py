import json
from optimizer import GeneticScheduleOptimizer
from visualization import ScheduleVisualizer, print_schedule


def load_config(path: str = "config.json") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_and_run():
    print("Загрузка конфигурации...")
    config = load_config()

    courses = list(config['COURSE_CONSTRAINTS'].keys())
    teachers = list(config['TEACHER_COURSES'].keys())
    time_slots = [f"{day} - {time}" for day in config['DAYS'] for time in config['TIMES']]
    classrooms = config['CLASSROOMS']
    student_groups = config['STUDENT_GROUPS']

    print("Создание оптимизатора и визуализатора...")
    visualizer = ScheduleVisualizer(config)
    optimizer = GeneticScheduleOptimizer(
        courses, classrooms, time_slots, teachers, student_groups, config
    )

    print("\nНачало оптимизации расписания...")
    best_schedule = optimizer.optimize_schedule(visualizer)

    print("\n" + "#" * 80 + "\nРЕЗУЛЬТАТ ОПТИМИЗАЦИИ\n" + "#" * 80)

    print_schedule(best_schedule, config)

    print("\nПостроение графиков...")
    visualizer.plot_optimization_progress()
    visualizer.plot_final_schedule_analysis(best_schedule, courses, teachers)
    visualizer.plot_group_schedules(best_schedule, student_groups)

    return best_schedule


if __name__ == '__main__':
    setup_and_run()