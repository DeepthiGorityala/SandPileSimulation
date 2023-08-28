import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define grid size, critical height and maximum topplings per iteration
GRID_SIZE = 100
CRITICAL_HEIGHT = 4
MAX_TOPPLINGS_PER_ITERATION = 20000

# Initialize grid with all zeros
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Defining some variables
iteration_number = 0
toppling_per_iteration = 0
total_topplings = 0
total_grains = []
average_height = []
total_topplings_per_iter = []
total_sites_toppled_per_iter = []

while True:
    # Selecting a random site on the grid
    u, v = np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)
    grid[u, v] += 1
    iteration_number = iteration_number + 1

    # Check for unstable sites and perform toppling
    unstable_sites = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i][j] >= CRITICAL_HEIGHT]

    if not unstable_sites:
        total_grains.append(np.sum(grid))
        average_height.append(np.sum(grid) / (GRID_SIZE ** 2))
        continue

    sites_toppled_per_iter = []
    sites_toppled = []
    topplings_per_iteration = 0
    while unstable_sites:
        topplings_per_iteration += len(unstable_sites)
        total_topplings += len(unstable_sites)
        sites_toppled = []

        for site in unstable_sites:
            # Reduce the number of grains in that site by 4
            grid[site[0], site[1]] -= 4
            if site[0] > 0:
                grid[site[0] - 1, site[1]] += 1
            if site[0] < GRID_SIZE - 1:
                grid[site[0] + 1, site[1]] += 1
            if site[1] > 0:
                grid[site[0], site[1] - 1] += 1
            if site[1] < GRID_SIZE - 1:
                grid[site[0], site[1] + 1] += 1
            sites_toppled.append(site)

        unstable_sites = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i][j] >= CRITICAL_HEIGHT]

        for site in unstable_sites:
            if site not in sites_toppled:
                sites_toppled_per_iter.append(site)

    total_grains.append(np.sum(grid))
    average_height.append(np.sum(grid) / (GRID_SIZE ** 2))
    total_topplings_per_iter.append(topplings_per_iteration)
    total_sites_toppled_per_iter.append(len(sites_toppled_per_iter))

    if max(total_topplings_per_iter) >= MAX_TOPPLINGS_PER_ITERATION:
        break


# Plot of average height on the board Vs Iteration number
plt.plot(average_height)
plt.xlabel('Iteration')
plt.ylabel('Average Height')
plt.show()

# To classify total topplings in each iteration into different beans of different sizes
s = []
Ps = []
m = 0
while max(total_topplings_per_iter) >= 2 ** m:
    m = m + 1
    if np.sum(total_topplings_per_iter.count(g) for g in total_topplings_per_iter if
              2 ** (m - 1) <= g <= (2 ** m) - 1) == 0:
        continue
    bw = (2 ** m) - (2 ** (m - 1))
    s.append(((2 ** (m - 1)) * (2 ** m)) ** 0.5)
    Ns = np.sum(
        total_topplings_per_iter.count(g) for g in total_topplings_per_iter if 2 ** (m - 1) <= g <= (2 ** m) - 1)
    Ps.append(Ns / (bw * total_topplings))
    Ps = [item for sublist in Ps for item in (sublist if isinstance(sublist, list) else [sublist])]


# print(Ps)
# print(s)


# Power law graph with fitting for total nuber of topplings in each iteration
def line_func(x, a, b):
    return a * x + b


x = np.log(s)
y = np.log(Ps)

popt, pcov = curve_fit(line_func, x, y)
a, b = popt
print(f"Equation of the line for total topplings per iteration: y = {a:.2f}x + {b:.2f}")

plt.plot(x, y, 'o', label='data')
plt.plot(x, line_func(x, *popt), 'r-', label='fit')
plt.legend()
plt.show()

# To classify total topplings in each iteration into different beans of different sizes
s = []
Ps = []
m = 0
while max(total_sites_toppled_per_iter) >= 2 ** m:
    m = m + 1
    if np.sum(total_sites_toppled_per_iter.count(g) for g in total_sites_toppled_per_iter if
              2 ** (m - 1) <= g <= (2 ** m) - 1) == 0:
        continue
    bw = (2 ** m) - (2 ** (m - 1))
    s.append(((2 ** (m - 1)) * (2 ** m)) ** 0.5)
    Ns = np.sum(total_sites_toppled_per_iter.count(g) for g in total_sites_toppled_per_iter if
                2 ** (m - 1) <= g <= (2 ** m) - 1)
    Ps.append(Ns / (bw * total_topplings))
    Ps = [item for sublist in Ps for item in (sublist if isinstance(sublist, list) else [sublist])]


# print(Ps)
# print(s)


# Power law graph with fitting for total number of sites(area) toppled in each iteration
def line_func(x, a, b):
    return a * x + b


x = np.log(s)
y = np.log(Ps)

popt, pcov = curve_fit(line_func, x, y)
a, b = popt
print(f"Equation of the line total sites toppled per iteration: y = {a:.2f}x + {b:.2f}")

plt.plot(x, y, 'o', label='data')
plt.plot(x, line_func(x, *popt), 'r-', label='fit')
plt.legend()
plt.show()
