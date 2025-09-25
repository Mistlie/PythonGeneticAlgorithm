import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from models import ScheduleNeuralNetwork


"""Генетический алгоритм с нейросетевым обучением"""
class GeneticScheduleOptimizer:

    def __init__(self, courses: List[str], classrooms: List[str], time_slots: List[str],
                 teachers: List[str], student_groups: List[str], ga_config: Dict):
        self.courses = courses
        self.classrooms = classrooms
        self.time_slots = time_slots
        self.teachers = teachers
        self.student_groups = student_groups
        self.teacher_courses = ga_config['TEACHER_COURSES']
        self.course_constraints = ga_config['COURSE_CONSTRAINTS']
        self.penalty_weights = ga_config['PENALTY_WEIGHTS']

        self.population_size = 150
        self.generation_count = 250

        self.input_size = len(courses) + len(teachers) + len(classrooms) + len(time_slots)
        self.nn = ScheduleNeuralNetwork(self.input_size)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001)

        self.course_to_idx = {c: i for i, c in enumerate(self.courses)}
        self.teacher_to_idx = {t: i for i, t in enumerate(self.teachers)}
        self.classroom_to_idx = {c: i for i, c in enumerate(self.classrooms)}
        self.time_slot_to_idx = {t: i for i, t in enumerate(self.time_slots)}
        self.training_data: List[Dict[str, Any]] = []

    def create_compact_features(self, schedule: List[Dict]) -> List[float]:
        if not schedule: return [0.0] * self.input_size
        course_hours = [0] * len(self.courses)
        teacher_load = [0] * len(self.teachers)
        classroom_usage = [0] * len(self.classrooms)
        time_slot_usage = [0] * len(self.time_slots)

        for lesson in schedule:
            course_hours[self.course_to_idx.get(lesson['course'], -1)] += 1
            teacher_load[self.teacher_to_idx.get(lesson['teacher'], -1)] += 1
            classroom_usage[self.classroom_to_idx.get(lesson['classroom'], -1)] += 1
            time_slot_usage[self.time_slot_to_idx.get(lesson['time_slot'], -1)] += 1

        total_lessons = len(schedule)
        features = []
        for counts in [course_hours, teacher_load, classroom_usage, time_slot_usage]:
            features.extend([c / total_lessons for c in counts])
        return features

    """Расчет приспособленности с учетом ограничений и предсказаний нейросети"""
    def calculate_fitness(self, chromosome: Dict) -> float:
        schedule = chromosome['schedule']
        feature_vector = self.create_compact_features(schedule)
        if len(feature_vector) != self.input_size:
            return 0.0

        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
        with torch.no_grad():
            quality_score, _ = self.nn(feature_tensor)

        penalty = self._calculate_penalties(schedule)
        constraint_satisfaction = self._calculate_constraint_satisfaction(schedule)

        base_fitness = quality_score.item() * constraint_satisfaction
        fitness = max(0.0, min(1.0, base_fitness * (1 - penalty)))

        chromosome['fitness'] = fitness
        return fitness

    """Расчет удовлетворения минимальным ограничениям по часам"""
    def _calculate_constraint_satisfaction(self, schedule: List[Dict]) -> float:
        if not schedule: return 0.0
        actual_hours = defaultdict(int)
        for lesson in schedule: actual_hours[lesson['course']] += 1

        satisfaction_scores = []
        for course, constraints in self.course_constraints.items():
            min_hours = constraints['min']
            actual = actual_hours.get(course, 0)
            satisfaction = (actual / min_hours) if actual < min_hours else (
                        1.0 + min((actual - min_hours) / (min_hours * 2), 0.5))
            satisfaction_scores.append(satisfaction)
        return sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0

    """Расчет штрафов"""
    def _calculate_penalties(self, schedule: List[Dict]) -> float:
        if not schedule: return 1.0

        teacher_load = defaultdict(int)
        classroom_usage = defaultdict(int)
        group_schedules = defaultdict(lambda: defaultdict(int))

        for lesson in schedule:
            day = lesson['time_slot'].split(' - ')[0]
            teacher_load[lesson['teacher']] += 1
            classroom_usage[lesson['classroom']] += 1
            group_schedules[lesson['group']][day] += 1

        penalties = {
            'teacher_load': self._teacher_load_penalty(teacher_load),
            'group_day_limit': self._group_day_limit_penalty(group_schedules),
            'schedule_gaps': self._schedule_gaps_penalty(schedule),
            'classroom_overload': self._classroom_overload_penalty(classroom_usage),
            'unused_timeslots': self._unused_timeslots_penalty(schedule)
        }

        total_penalty = sum(penalties[key] * self.penalty_weights[key] for key in penalties)
        total_weight = sum(self.penalty_weights.values())

        return min(total_penalty / total_weight, 1.0) if total_weight > 0 else 0.0

    """Штраф за неравномерную нагрузку преподавателей"""
    def _teacher_load_penalty(self, teacher_load: Dict[str, int]) -> float:
        if not teacher_load: return 0.0
        load_values = list(teacher_load.values())
        load_std = np.std(load_values) if len(load_values) > 1 else 0
        max_load = max(load_values) if load_values else 1
        return load_std / max_load if max_load > 0 else 0.0

    """Штраф за превышение 6 пар в день для групп"""
    def _group_day_limit_penalty(self, group_schedules: Dict[str, Dict[str, int]]) -> float:
        penalty = 0
        for _, days in group_schedules.items():
            for _, count in days.items():
                if count > 6:
                    penalty += (count - 6) / 6
        return penalty

    """Штраф за окна в расписании групп"""
    def _schedule_gaps_penalty(self, schedule: List[Dict]) -> float:
        group_schedules_detail = defaultdict(list)
        for lesson in schedule:
            day = lesson['time_slot'].split(' - ')[0]
            slot_num = self.time_slot_to_idx.get(lesson['time_slot'], -1)
            if slot_num != -1:
                group_schedules_detail[(lesson['group'], day)].append(slot_num)

        gap_penalty = 0
        for _, slots in group_schedules_detail.items():
            if len(slots) > 1:
                slots.sort()
                gaps = sum(slots[i + 1] - slots[i] - 1 for i in range(len(slots) - 1))
                if gaps > 0:
                    gap_penalty += min(gaps / len(slots), 1.0)
        return gap_penalty

    """Штраф за перегрузку аудиторий"""
    def _classroom_overload_penalty(self, classroom_usage: Dict[str, int]) -> float:
        if not classroom_usage: return 0.0
        max_usage = max(classroom_usage.values())
        max_limit = len(self.time_slots) * 0.7  # 70% загрузка
        return (max_usage - max_limit) / len(self.time_slots) if max_usage > max_limit else 0.0

    """Штраф за неиспользованные временные слоты"""
    def _unused_timeslots_penalty(self, schedule: List[Dict]) -> float:
        used_slots = len(set(lesson['time_slot'] for lesson in schedule))
        return 1.0 - (used_slots / len(self.time_slots))

    """Проверка валидности занятия"""
    def is_valid_lesson(self, lesson: Dict, existing_schedule: List[Dict]) -> bool:
        has_conflict = any(
            l['time_slot'] == lesson['time_slot'] and (
                    l['teacher'] == lesson['teacher'] or
                    l['classroom'] == lesson['classroom'] or
                    l['group'] == lesson['group']
            )
            for l in existing_schedule
        )
        if has_conflict:
            return False

        day = lesson['time_slot'].split(' - ')[0]
        group_day_lessons = sum(1 for l in existing_schedule
                                if l['group'] == lesson['group'] and l['time_slot'].startswith(day))

        return group_day_lessons < 6

    def _prepare_nn_batch(self, data: List[Dict], batch_size: Optional[int]) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not data:
            return None, None

        batch = random.sample(data, min(batch_size, len(data))) if batch_size else data

        features = []
        targets = []

        for item in batch:
            if item['features'] and len(item['features']) == self.input_size:
                features.append(item['features'])
                targets.append([min(max(item['fitness'], 0.0), 1.0)])

        if not features:
            return None, None

        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        return features_tensor, targets_tensor

    """Обучение на батче данных"""
    def train_batch(self, data: List[Dict], batch_size: int) -> Optional[float]:
        features_tensor, targets_tensor = self._prepare_nn_batch(data, batch_size)
        if features_tensor is None:
            return None

        self.optimizer.zero_grad()
        quality_pred, _ = self.nn(features_tensor)
        loss = nn.MSELoss()(quality_pred, targets_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    """Валидация на батче данных"""
    def validate_batch(self, data: List[Dict]) -> Optional[float]:
        features_tensor, targets_tensor = self._prepare_nn_batch(data, len(data))
        if features_tensor is None:
            return None

        with torch.no_grad():
            quality_pred, _ = self.nn(features_tensor)
            loss = nn.MSELoss()(quality_pred, targets_tensor)

        return loss.item()

    """Улучшенное обучение с валидацией"""
    def enhanced_training(self, batch_size=32) -> Tuple[Optional[float], Optional[float]]:
        if len(self.training_data) < batch_size * 2:
            return None, None

        train_size = int(0.8 * len(self.training_data))
        train_data = self.training_data[:train_size]
        val_data = self.training_data[train_size:]

        train_loss = self.train_batch(train_data, batch_size)
        val_loss = self.validate_batch(val_data)

        return train_loss, val_loss

    """Создание случайной хромосомы (расписания)"""
    def create_chromosome(self) -> Dict:
        chromosome = {
            'schedule': [],
            'fitness': 0,
            'course_constraints': self.course_constraints.copy()
        }

        course_hours = {
            course: constraints['min'] + random.randint(0, 3)
            for course, constraints in self.course_constraints.items()
        }

        for course, hours in course_hours.items():
            for _ in range(hours):
                attempts = 0
                max_attempts = 500

                while attempts < max_attempts:
                    available_teachers = self.get_available_teachers(course)
                    if not available_teachers:
                        break

                    lesson = {
                        'course': course,
                        'teacher': random.choice(available_teachers),
                        'classroom': random.choice(self.classrooms),
                        'time_slot': random.choice(self.time_slots),
                        'group': random.choice(self.student_groups)
                    }

                    if self.is_valid_lesson(lesson, chromosome['schedule']):
                        chromosome['schedule'].append(lesson)
                        break

                    attempts += 1

        return chromosome

    """Проверка на конфликты"""
    def check_hard_constraints(self, schedule: List[Dict]) -> int:
        violations = 0
        for i, lesson1 in enumerate(schedule):
            for j, lesson2 in enumerate(schedule[i + 1:], i + 1):
                if (lesson1['time_slot'] == lesson2['time_slot'] and
                        (lesson1['teacher'] == lesson2['teacher'] or
                         lesson1['classroom'] == lesson2['classroom'] or
                         lesson1['group'] == lesson2['group'])):
                    violations += 1
        return violations

    """Комплексная валидация расписания"""
    def validate_schedule(self, schedule: List[Dict]) -> bool:
        return bool(schedule) and self.check_hard_constraints(schedule) == 0

    """Улучшенный кроссовер с сохранением валидности"""
    def improved_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child_schedule = []
        combined = parent1['schedule'] + parent2['schedule']
        random.shuffle(combined)

        used_slots = set()
        for lesson in combined:
            keys = [
                (lesson['time_slot'], lesson['classroom']),
                (lesson['time_slot'], lesson['teacher']),
                (lesson['time_slot'], lesson['group'])
            ]

            if not any(key in used_slots for key in keys):
                if self.is_valid_lesson(lesson, child_schedule):
                    child_schedule.append(lesson)
                    for key in keys:
                        used_slots.add(key)

        return {'schedule': child_schedule, 'fitness': 0, 'course_constraints': self.course_constraints.copy()}

    """Кроссовер двух родителей"""
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        return self.improved_crossover(parent1, parent2)

    """Мутация хромосомы"""
    def mutate(self, chromosome: Dict) -> Dict:
        mutated = chromosome.copy()
        mutated_schedule = mutated['schedule'].copy()

        mutation_rate = 0.3
        feature_vector = self.create_compact_features(mutated_schedule)
        if len(feature_vector) == self.input_size:
            feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            with torch.no_grad():
                _, ga_params = self.nn(feature_tensor)
                mutation_rate = min(0.5, ga_params[0, 0].item() * 2)

        if random.random() < mutation_rate and mutated_schedule:
            mutation_type = random.choice(['add', 'remove', 'modify'])

            if mutation_type == 'add' and len(mutated_schedule) < len(self.time_slots) * len(self.student_groups):
                course = random.choice(self.courses)
                available_teachers = self.get_available_teachers(course)
                if available_teachers:
                    new_lesson = {
                        'course': course,
                        'teacher': random.choice(available_teachers),
                        'classroom': random.choice(self.classrooms),
                        'time_slot': random.choice(self.time_slots),
                        'group': random.choice(self.student_groups)
                    }
                    if self.is_valid_lesson(new_lesson, mutated_schedule):
                        mutated_schedule.append(new_lesson)

            elif mutation_type == 'remove':
                mutated_schedule.pop(random.randint(0, len(mutated_schedule) - 1))

            elif mutation_type == 'modify':
                lesson_idx = random.randint(0, len(mutated_schedule) - 1)
                lesson_to_modify = mutated_schedule.pop(lesson_idx)

                attempts = 0
                while attempts < 50:
                    modified_lesson = lesson_to_modify.copy()

                    modified_lesson['time_slot'] = random.choice(self.time_slots)
                    modified_lesson['classroom'] = random.choice(self.classrooms)

                    if self.is_valid_lesson(modified_lesson, mutated_schedule):
                        mutated_schedule.append(modified_lesson)
                        break

                    attempts += 1

                if attempts >= 50:
                    mutated_schedule.insert(lesson_idx, lesson_to_modify)

        mutated['schedule'] = mutated_schedule
        return mutated

    """Турнирный отбор"""
    def select_parents(self, population: List[Dict]) -> Tuple[Dict, Dict]:
        valid_population = [chromo for chromo in population if chromo['schedule']]
        if len(valid_population) < 2:
            return self.create_chromosome(), self.create_chromosome()

        tournament_size = max(2, min(5, len(valid_population) // 10))

        def run_tournament():
            tournament = random.sample(valid_population, tournament_size)
            return max(tournament, key=lambda x: x.get('fitness', 0))

        parent1 = run_tournament()
        parent2 = run_tournament()

        return parent1, parent2

    """Алгоритм оптимизации расписания"""
    def optimize_schedule(self, visualizer: 'ScheduleVisualizer' = None) -> Dict:
        print("Инициализация популяции...")
        population = [self.create_chromosome() for _ in range(self.population_size)]

        best_schedule = None
        best_fitness = 0

        for generation in range(1, self.generation_count + 1):
            total_fitness = 0
            total_penalty = 0
            total_satisfaction = 0
            valid_chromosomes = 0

            for chromosome in population:
                if chromosome['schedule'] and self.validate_schedule(chromosome['schedule']):
                    fitness = self.calculate_fitness(chromosome)

                    total_fitness += fitness
                    total_penalty += self._calculate_penalties(chromosome['schedule'])
                    total_satisfaction += self._calculate_constraint_satisfaction(chromosome['schedule'])
                    valid_chromosomes += 1

                    features = self.create_compact_features(chromosome['schedule'])
                    if features and len(features) == self.input_size:
                        self.training_data.append({'features': features, 'fitness': fitness})

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_schedule = chromosome.copy()
                else:
                    chromosome['fitness'] = 0

            if not population:
                break

            avg_fitness = total_fitness / valid_chromosomes if valid_chromosomes > 0 else 0

            train_loss, val_loss = (None, None)
            if generation % 20 == 0 or generation == self.generation_count:
                train_loss, val_loss = self.enhanced_training()

            loss_info = ""
            if train_loss is not None:
                loss_info += f"Потери: train={train_loss:.4f}"
                if val_loss is not None:
                    loss_info += f", val={val_loss:.4f}"

            print(f"Поколение {generation}: Лучшая приспособленность = {best_fitness:.4f}, "
                  f"Средняя = {avg_fitness:.4f}. {loss_info}")

            if visualizer and valid_chromosomes > 0:
                visualizer.update_training_history(
                    generation,
                    avg_fitness,
                    total_penalty / valid_chromosomes,
                    total_satisfaction / valid_chromosomes,
                    train_loss,
                    val_loss
                )

            new_population = []
            elite_size = max(1, int(self.population_size * 0.15))

            elites = sorted([c for c in population if c['schedule']],
                            key=lambda x: x['fitness'], reverse=True)[:elite_size]
            new_population.extend(elites)

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                if child['schedule'] and self.validate_schedule(child['schedule']):
                    new_population.append(child)
                else:
                    new_population.append(self.create_chromosome())

            population = new_population

        return best_schedule if best_schedule else self.create_chromosome()

    """Получение преподавателей, которые могут вести предмет"""
    def get_available_teachers(self, course: str) -> List[str]:
        return [t for t, c in self.teacher_courses.items() if course in c]