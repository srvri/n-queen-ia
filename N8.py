import random
import math
import time
import copy
from queue import PriorityQueue


# تابع محاسبه تعداد تعارضات
def count_conflicts(board):
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j]:  # تعارض در ستون
                conflicts += 1
            if abs(board[i] - board[j]) == abs(i - j):  # تعارض در قطر
                conflicts += 1
    return conflicts


# تابع چاپ صفحه شطرنج
def print_board(board):
    n = len(board)
    for i in range(n):
        row = ['.'] * n
        row[board[i]] = 'Q'
        print(' '.join(row))
    print()


# 1. Hill Climbing (Random Start)
def hill_climbing(n):
    board = [random.randint(0, n - 1) for _ in range(n)]
    steps = 0
    while True:
        current_conflicts = count_conflicts(board)
        if current_conflicts == 0:
            return board, steps
        neighbors = []
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    neighbors.append((count_conflicts(neighbor), neighbor))
        neighbors.sort()
        if neighbors[0][0] >= current_conflicts:
            break  # گیر کردن در بهینه محلی
        board = neighbors[0][1]
        steps += 1
    return board, steps


# 2. Hill Climbing with Sideways Moves
def hill_climbing_sideways(n, max_sideways=100):
    board = [random.randint(0, n - 1) for _ in range(n)]
    steps = 0
    sideways = 0
    while True:
        current_conflicts = count_conflicts(board)
        if current_conflicts == 0:
            return board, steps
        neighbors = []
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    neighbors.append((count_conflicts(neighbor), neighbor))
        neighbors.sort()
        if neighbors[0][0] > current_conflicts:
            break
        elif neighbors[0][0] == current_conflicts:
            if sideways >= max_sideways:
                break
            sideways += 1
        else:
            sideways = 0
        board = neighbors[0][1]
        steps += 1
    return board, steps


# 3. Stochastic Hill Climbing
def stochastic_hill_climbing(n):
    board = [random.randint(0, n - 1) for _ in range(n)]
    steps = 0
    while True:
        current_conflicts = count_conflicts(board)
        if current_conflicts == 0:
            return board, steps
        neighbors = []
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    conflicts = count_conflicts(neighbor)
                    if conflicts <= current_conflicts:
                        neighbors.append(neighbor)
        if not neighbors:
            break
        board = random.choice(neighbors)
        steps += 1
    return board, steps


# 4. Simulated Annealing
def simulated_annealing(n, initial_temp=1000, cooling_rate=0.995):
    board = [random.randint(0, n - 1) for _ in range(n)]
    temp = initial_temp
    steps = 0
    while temp > 0.1:
        current_conflicts = count_conflicts(board)
        if current_conflicts == 0:
            return board, steps
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if j != board[i]:
            neighbor = board.copy()
            neighbor[i] = j
            neighbor_conflicts = count_conflicts(neighbor)
            delta = neighbor_conflicts - current_conflicts
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                board = neighbor
            steps += 1
        temp *= cooling_rate
    return board, steps


# 5. Genetic Algorithm
def genetic_algorithm(n, population_size=100, max_generations=1000):
    def create_individual():
        return [random.randint(0, n - 1) for _ in range(n)]

    def crossover(parent1, parent2):
        point = random.randint(1, n - 2)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(individual):
        i = random.randint(0, n - 1)
        individual[i] = random.randint(0, n - 1)
        return individual

    population = [create_individual() for _ in range(population_size)]
    steps = 0
    for _ in range(max_generations):
        population = sorted(population, key=count_conflicts)
        if count_conflicts(population[0]) == 0:
            return population[0], steps
        new_population = population[:10]  # نگه‌داشتن 10 فرد برتر
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population[:50], k=2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        steps += 1
    return population[0], steps


# 6. Min-Conflicts (رفع خطا)
def min_conflicts(n, max_steps=1000):
    board = [random.randint(0, n - 1) for _ in range(n)]
    steps = 0
    for _ in range(max_steps):
        if count_conflicts(board) == 0:
            return board, steps
        conflicts = []
        for i in range(n):
            for j in range(n):
                temp_board = board.copy()
                temp_board[i] = j
                conflicts.append((i, j, count_conflicts(temp_board)))
        max_conflict_row = max(conflicts, key=lambda x: x[2])[0]
        min_conflict_pos = min([(j, count_conflicts([board[k] if k != max_conflict_row else j for k in range(n)]))
                                for j in range(n)], key=lambda x: x[1])[0]
        board[max_conflict_row] = min_conflict_pos
        steps += 1
    return board, steps


# 7. Random Restart Hill Climbing
def random_restart_hill_climbing(n, max_restarts=10):
    steps = 0
    for _ in range(max_restarts):
        board, local_steps = hill_climbing(n)
        steps += local_steps
        if count_conflicts(board) == 0:
            return board, steps
    return board, steps


# 8. Tabu Search
def tabu_search(n, tabu_size=50, max_steps=1000):
    board = [random.randint(0, n - 1) for _ in range(n)]
    tabu_list = []
    steps = 0
    for _ in range(max_steps):
        if count_conflicts(board) == 0:
            return board, steps
        neighbors = []
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    if neighbor not in tabu_list:
                        neighbors.append((count_conflicts(neighbor), neighbor))
        if not neighbors:
            break
        neighbors.sort()
        board = neighbors[0][1]
        tabu_list.append(board.copy())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        steps += 1
    return board, steps


# 9. A* Search
def a_star_search(n):
    initial_board = [random.randint(0, n - 1) for _ in range(n)]
    queue = PriorityQueue()
    queue.put((count_conflicts(initial_board), 0, initial_board))
    visited = set()
    steps = 0
    while not queue.empty():
        conflicts, cost, board = queue.get()
        if conflicts == 0:
            return board, steps
        board_tuple = tuple(board)
        if board_tuple in visited:
            continue
        visited.add(board_tuple)
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    queue.put((count_conflicts(neighbor) + cost + 1, cost + 1, neighbor))
        steps += 1
    return board, steps


# 10. Greedy Best-First Search
def greedy_best_first_search(n):
    board = [random.randint(0, n - 1) for _ in range(n)]
    steps = 0
    while True:
        current_conflicts = count_conflicts(board)
        if current_conflicts == 0:
            return board, steps
        neighbors = []
        for i in range(n):
            for j in range(n):
                if j != board[i]:
                    neighbor = board.copy()
                    neighbor[i] = j
                    neighbors.append((count_conflicts(neighbor), neighbor))
        neighbors.sort()
        board = neighbors[0][1]
        steps += 1
        if steps > 1000:  # جلوگیری از حلقه بی‌نهایت
            break
    return board, steps


# اجرای همه ابتکارها برای n=4
def main():
    n = 8  # تغییر تعداد وزیرها به ۸
    heuristics = [
        ("Hill Climbing", hill_climbing),
        ("Hill Climbing with Sideways", hill_climbing_sideways),
        ("Stochastic Hill Climbing", stochastic_hill_climbing),
        ("Simulated Annealing", simulated_annealing),
        ("Genetic Algorithm", genetic_algorithm),
        ("Min-Conflicts", min_conflicts),
        ("Random Restart Hill Climbing", random_restart_hill_climbing),
        ("Tabu Search", tabu_search),
        ("A* Search", a_star_search),
        ("Greedy Best-First Search", greedy_best_first_search)
    ]

    results = []
    for name, heuristic in heuristics:
        start_time = time.time()
        board, steps = heuristic(n)
        end_time = time.time()
        conflicts = count_conflicts(board)
        results.append({
            "name": name,
            "board": board,
            "steps": steps,
            "conflicts": conflicts,
            "time": end_time - start_time
        })
        print(f"\n{name}:")
        print(f"Board: {board}")
        print(f"Conflicts: {conflicts}")
        print(f"Steps: {steps}")
        print(f"Time: {end_time - start_time:.6f} seconds")
        if conflicts == 0:
            print("Solution found:")
            print_board(board)

    # تحلیل نتایج
    print("\nتحلیل نتایج:")
    best_heuristic = min(results, key=lambda x: (x["conflicts"], x["steps"], x["time"]))
    print(f"بهترین ابتکار: {best_heuristic['name']}")
    print(
        f"دلیل: این ابتکار با {best_heuristic['steps']} گام و {best_heuristic['time']:.6f} ثانیه، تعداد تعارضات {best_heuristic['conflicts']} را به دست آورد.")
    print("تحلیل جزئی:")
    for result in results:
        print(f"{result['name']}:")
        print(f"  - تعداد گام‌ها: {result['steps']}")
        print(f"  - زمان اجرا: {result['time']:.6f} ثانیه")
        print(f"  - تعداد تعارضات: {result['conflicts']}")
        if result["conflicts"] == 0:
            print("  - این ابتکار به راه‌حل بهینه رسیده است.")
        else:
            print("  - این ابتکار به راه‌حل بهینه نرسیده است.")



if __name__ == "__main__":
    main()
