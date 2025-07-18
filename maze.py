import numpy as np
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, size: tuple, seed = None):
        if seed:
            np.random.seed(seed)
        self.size = size
        self.start = (0, 5)
        self.end = (self.size[0]-1, 5)
        self.current = self.end
        self.squares = np.random.random(size) + 0.05
        self.build_walls()
        self.distance_map = None
        self.shortest_way = None
        self.generated = False
        self.distance_frames = []
        self.way_frames = []


    def __repr__(self):
        return np.array2string(self.squares)
        
    
    def build_walls(self):
        self.squares[0::2,:] = 0
        self.squares[:,0::2] = 0
        x_start, y_start = self.start
        x_end, y_end = self.end
        self.squares[self.start] = np.random.random()
        self.squares[self.end] = np.random.random()


    def random_neighbor(self, cell):
        directions = ['up','down','right','left']
        np.random.shuffle(directions)
        for direction in directions:
            if direction == 'up':
                if cell[1] == 1:
                    continue
            if direction == 'down':
                if cell[1] == self.size[1] - 2:
                    continue
            if direction == 'right':
                if cell[0] == self.size[0] - 2:
                    continue
            if direction == 'left':
                if cell[0] == 1:
                    continue
            return direction
        return None

        
    def generate(self):
        x = np.random.randint(0,self.size[0]//2)*2+1
        y = np.random.randint(0,self.size[1]//2)*2+1
        color = self.squares[x,y]
        walls = self.squares[1:-1,1:-1] == 0
        color_path = self.squares[1:-1,1:-1] == color
        states = [np.copy(self.squares)]
        i = 0

        while not (walls | color_path).all() and i < 10000:
            neighbor = self.random_neighbor((x,y))
            color = self.squares[x,y]
            if neighbor == 'up' and self.squares[x,y-2] != color:
                other_color = self.squares[x,y-2]
                self.squares[x,y-1] = color
                self.squares[self.squares == other_color] = color
            if neighbor == 'down' and self.squares[x,y+2] != color:
                other_color = self.squares[x,y+2]
                self.squares[x,y+1] = color
                self.squares[self.squares == other_color] = color
            if neighbor == 'right' and self.squares[x+2,y] != color:
                other_color = self.squares[x+2,y]
                self.squares[x+1,y] = color
                self.squares[self.squares == other_color] = color
            if neighbor == 'left' and self.squares[x-2,y] != color:
                other_color = self.squares[x-2,y]
                self.squares[x-1,y] = color
                self.squares[self.squares == other_color] = color
            walls = self.squares[1:-1,1:-1] == 0
            color_path = self.squares[1:-1,1:-1] == color
            x = np.random.randint(0,self.size[0]//2)*2+1
            y = np.random.randint(0,self.size[1]//2)*2+1
            state = np.copy(self.squares)
            state[state == 0] = np.nan
            states.append(state)
        self.squares[self.start] = self.squares[1,1]
        self.squares[self.end] = self.squares[1,1]
        state = np.copy(self.squares)
        state[state == 0] = np.nan
        states.append(state)
        self.generated = True
        return states


    def break_random_walls(self, pct_wall_to_destroy = 0.05):
        walls = self.squares == 0 
        chance_destroy = np.random.random(self.size) < pct_wall_to_destroy
        walls_to_destroy = (walls & chance_destroy)
        for i,row in enumerate(walls_to_destroy[1:-1,1:-1]):
            for j,is_wall in enumerate(row):
                x = i+1
                y = j+1
                if is_wall:
                    if (((self.squares[x+1,y] == 0)
                    and (self.squares[x-1,y] == 0))
                    and ((self.squares[x,y+1] != 0)
                    and (self.squares[x,y-1] != 0))):
                        self.squares[x,y] = self.squares[1,1]
                    if (((self.squares[x,y+1] == 0)
                    and (self.squares[x,y-1] == 0))
                    and ((self.squares[x+1,y] != 0)
                    and (self.squares[x-1,y] != 0))):
                        self.squares[x,y] = self.squares[1,1]


    def find_neighbors(self, point):
        neighbors = []
        x, y = point[0], point[1]
        if (x < self.size[0] - 1 and 
            self.squares[x+1, y] != 0):
            neighbors.append((x+1, y))
        if (x > 0 and 
            self.squares[x-1, y] != 0):
            neighbors.append((x-1, y))
        if (y < self.size[1] - 1 and 
            self.squares[x, y+1] != 0):
            neighbors.append((x, y+1))
        if (y > 0 and 
            self.squares[x, y-1] != 0):
            neighbors.append((x, y-1))
        return neighbors

    def find_way_A_star(self):
        def h(point):
            return (abs(point[0]-self.end[0])+
                    abs(point[1]-self.end[1]))
        # chaque point dans la liste est stocké sous la forme point: [f(n), g(n), point parent]
        open_list = {self.start: [0 + h(self.start), 0, None]}
        closed_list = {}
        while self.end and open_list:
            #On cherche le point de la open_list avec la fonction f la plus basse
            min_f = np.inf
            min_f_point = None
            for point in open_list:
                if open_list[point][0] < min_f:
                    min_f = open_list[point][0]
                    min_f_point = point
            #On ajoute ce point a la closed list avec son poid g
            closed_list[min_f_point] = open_list[min_f_point]
            del(open_list[min_f_point])
            point_neighbors = self.find_neighbors(min_f_point)
            for point in point_neighbors:
                if point in closed_list: continue
                g = closed_list[min_f_point][1]
                if point not in open_list:
                    open_list[point] = [g + h(point), g, min_f_point]
                elif g < open_list[point][1]:
                    open_list[point] = [g + h(point), g, min_f_point]
        closed_list[self.end] = open_list[self.end]
        current = self.end
        path = [current]
        while closed_list[current][2]:
            path.append(closed_list[current][2])
            current = closed_list[current][2]
        mat_path = np.full(self.size,np.nan)
        for point in path:
            mat_path[point] = 1
        self.shortest_way = mat_path
        self.way_frames.append(mat_path)
        return mat_path


    def plot(self):
        plt.tight_layout()
        plt.pcolormesh(np.zeros(self.size), cmap = 'gray')
        plot_mat = np.copy(self.squares)
        plot_mat[plot_mat == 0] = np.nan
        plt.pcolormesh(plot_mat, cmap = 'terrain_r')


    def plot_way(self):
        way = self.find_way_A_star()
        plt.pcolormesh(way, cmap = 'RdPu_r')
