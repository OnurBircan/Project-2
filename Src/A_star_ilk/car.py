import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heapq import heappush, heappop
import random

class AStarPathPlanning:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape
        self.open_list = []
        self.closed_set = set()
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start)}

    def heuristic(self, position):
        """Manhattan distance heuristic"""
        return abs(position[0] - self.goal[0]) + abs(position[1] - self.goal[1])

    def neighbors(self, position):
        """Get valid neighbors of a position"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for d in directions:
            neighbor = (position[0] + d[0], position[1] + d[1])
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                if self.grid[neighbor] == 0:  # Walkable cell
                    yield neighbor

    def reconstruct_path(self):
        """Reconstruct path from start to goal"""
        path = []
        current = self.goal
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def find_path(self):
        """Execute A* algorithm"""
        heappush(self.open_list, (self.f_score[self.start], self.start))
        while self.open_list:
            _, current = heappop(self.open_list)

            if current == self.goal:
                return self.reconstruct_path()

            self.closed_set.add(current)
            for neighbor in self.neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.g_score[current] + 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor)
                    heappush(self.open_list, (self.f_score[neighbor], neighbor))

        return None  # No path found

def generate_random_grid(rows, cols, obstacle_ratio=0.3):
    """Generate a grid with random obstacles"""
    grid = np.zeros((rows, cols), dtype=int)
    num_obstacles = int(rows * cols * obstacle_ratio)
    for _ in range(num_obstacles):
        x, y = random.randint(0, rows - 1), random.randint(0, cols - 1)
        grid[x, y] = 1
    return grid

def get_random_start_goal(grid):
    """Generate random start and goal positions"""
    rows, cols = grid.shape
    while True:
        start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        goal = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        if grid[start] == 0 and grid[goal] == 0 and start != goal:
            return start, goal

def animate_simulation(grid, path, start, goal):
    """Animate the car moving along the path"""
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_copy = np.copy(grid)

    # Colormap and color boundaries
    cmap = plt.cm.colors.ListedColormap(["white", "black", "green", "red", "blue"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Mark start and goal in the grid
    grid_copy[start] = 3  # Red for start
    grid_copy[goal] = 4   # Blue for goal

    img = ax.imshow(grid_copy, cmap=cmap, norm=norm, origin="upper")

    # Add legend for colors
    legend_labels = [
        "Empty (White)",
        "Obstacle (Black)",
        "Path (Green)",
        "Start (Red)",
        "Goal (Blue)"
    ]
    legend_colors = ["white", "black", "green", "red", "blue"]
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, legend_colors)]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)

    def update(frame):
        """Update the grid for animation"""
        if frame < len(path):
            pos = path[frame]
            if pos != start and pos != goal:
                grid_copy[pos] = 2  # Mark as green
            img.set_data(grid_copy)
        return img,

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), interval=50, repeat=False
    )
    plt.show()

# Simulation Parameters
rows, cols = 64, 64

# Generate Random Grid and Obstacles
grid = generate_random_grid(rows, cols, obstacle_ratio=0.2)
start, goal = get_random_start_goal(grid)

# Run A* Algorithm
astar = AStarPathPlanning(grid, start, goal)
path = astar.find_path()

if path:
    print(f"Path found: {path}")
    animate_simulation(grid, path, start, goal)
else:
    print("No path found!")