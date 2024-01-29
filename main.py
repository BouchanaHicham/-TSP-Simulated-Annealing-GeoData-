import random
import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id, x, y):
        self.x = float(x)
        self.y = float(y)
        self.id = int(id)

class Solution:
    # ----------------------------------------- [ Initialization ] -----------------------------------------
    def __init__(self, node_list):
        self.Solution = node_list

        slt_representation = []           
        for i in range(0, len(node_list)):
            slt_representation.append(self.Solution[i].id)
        self.slt_representation = slt_representation

        distance = 0
        for j in range(1, len(self.slt_representation) - 1):
            distance += matrix[self.slt_representation[j]-1][self.slt_representation[j + 1]-1]
        self.cost = distance

        self.fitness_value = 1 / self.cost
    # -------------------------------------- [ Neighborhood Generation ] --------------------------------------
    def generate_neighbor(self):
        neighbor = self.Solution.copy()
        # Implement logic to modify the neighbor solution (e.g., swap two cities)
        index1, index2 = random.sample(range(1, 58), 2)
        neighbor[index1], neighbor[index2] = neighbor[index2], neighbor[index1]
        return Solution(neighbor)

# Function to create a random list of cities for the initial solution
def create_random_list(n_list):
    start = n_list[0]
    temp = n_list[1:]
    temp = random.sample(temp, len(temp))
    temp.insert(0, start)
    temp.append(start)
    return temp

# Function to create a distance matrix based on node coordinates
def create_distance_matrix(node_list):
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(0, len(matrix)-1):
        for j in range(0, len(matrix[0])-1):
            matrix[node_list[i].id][node_list[j].id] = math.sqrt(pow((node_list[i].x - node_list[j].x), 2) + pow((node_list[i].y - node_list[j].y), 2))
    return matrix

# Cooling schedule function for simulated annealing
def cooling_schedule(iteration):
    return initial_temperature * cooling_rate**iteration

# Simulated Annealing algorithm
def simulated_annealing(initial_solution, temperature, cooling_rate, num_iterations, print_interval=100):
    current_solution = initial_solution
    best_solution = current_solution

    print("Iteration\tCurrent Cost\t\tNeighbor Cost\t\tProbability\tAcceptance\tBest Cost")

    for iteration in range(num_iterations):
        neighbor_solution = current_solution.generate_neighbor()

        current_cost = current_solution.cost
        neighbor_cost = neighbor_solution.cost

        energy_difference = current_cost - neighbor_cost
        # ---------------------------------- [ Choice of a neighboring solution ] ----------------------------------
        # Accept the neighbor if it's better or with a certain probability
        probability = math.exp((energy_difference) / temperature)
        acceptance = neighbor_cost < current_cost or random.uniform(0, 1) < probability

        if acceptance:
            current_solution = neighbor_solution

        # Evaluation: Update the best solution if the current solution is better
        if current_solution.cost < best_solution.cost:
            best_solution = current_solution

        # Print information every 'print_interval' iterations
        if (iteration + 1) % print_interval == 0 or iteration == num_iterations - 1:
            print(f"{iteration + 1}\t\t{current_cost:.4f}\t\t{neighbor_cost:.4f}\t\t{probability:.4f}\t\t{acceptance}\t\t{best_solution.cost:.4f}")

        temperature = cooling_schedule(iteration)

    return best_solution

# Read data from the file
file_name = "data_set"
dataset = []

with open(file_name, "r") as f:
    for line in f:
        new_line = line.strip()
        new_line = new_line.split(" ")
        id, y, x = new_line[0], new_line[1], new_line[2]
        dataset.append(Node(id=id, x=x, y=y))

N = 58
matrix = create_distance_matrix(dataset)

# Set parameters for Simulated Annealing
initial_temperature = 1000
cooling_rate = 0.99
num_iterations_sa = 1000

# Run Simulated Annealing
initial_solution_sa = Solution(create_random_list(dataset))
final_solution_sa = simulated_annealing(initial_solution_sa, initial_temperature, cooling_rate, num_iterations_sa, print_interval=100)

# Plot the best path
x_list = [node.x for node in final_solution_sa.Solution]
y_list = [node.y for node in final_solution_sa.Solution]

fig, ax = plt.subplots()
plt.scatter(x_list, y_list)
ax.plot(x_list, y_list, '--', lw=0.1, color='black', ms=10)
ax.set_title("TSP using Simulated Annealing")
ax.set_xlabel("X-coordinate")
ax.set_ylabel("Y-coordinate")



# Print the best path in terms of city names
cities = ["Algiers", "Oran", "Constantine", "Blida", "Batna", "Sétif", "Djelfa", "Annaba", "Sidi Bel Abbès", "Biskra",
"Tébessa", "Tiaret", "Bejaïa", "Tlemcen", "Bordj Bou Arreridj", "Béchar", "Skikda", "Souk Ahras", "Chlef",
"M’Sila", "Mostaganem", "Médéa", "Tizi Ouzou", "El Oued", "Laghouat", "Ouargla", "Jijel", "Relizane", "Saïda",
"Guelma", "Ghardaïa", "Mascara", "Khenchela", "Oum el Bouaghi", "El Bayadh", "Tamanrasset", "Aïn Temouchent",
"Tissemsilt", "Bouira", "Adrar", "Tindouf", "Boumerdes", "El Golea", "Touggourt", "Timimoun", "I-n-Salah",
"El Tarf", "Tipasa", "Illizi", "Bordj Mokhtar", "Naama", "Djanet", "Beni Abbès", "In Guezzam", "Aïn Defla",
"Mila", "Ouled Djellal", "El Meghaïer"]

city_array = [node.id for node in final_solution_sa.Solution]

Shortest_city_path_list = []
start_city = "Biskra"
start_city_index = cities.index(start_city)

Shortest_city_path_list = cities[start_city_index:] + cities[:start_city_index]
print("----------------------------------------------------------")
print("Shortest Path using Simulated Annealing:")
print("----------------------------------------")
print(Shortest_city_path_list)
print("----------------------------------------------------------")
plt.show()