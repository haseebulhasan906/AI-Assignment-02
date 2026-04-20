import pygame
import heapq
import math
import random
import time

COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GREEN': (0, 200, 0),
    'RED': (200, 0, 0),
    'BLUE': (100, 150, 255),
    'YELLOW': (255, 255, 100),
    'GRAY': (180, 180, 180),
    'DARK_GRAY': (60, 60, 60),
}

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.is_wall = False
        self.is_start = False
        self.is_goal = False
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.visited = False
        self.in_frontier = False
        
    def reset(self):
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.visited = False
        self.in_frontier = False

class PathfindingAgent:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[Node(r, c) for c in range(cols)] for r in range(rows)]
        self.start = None
        self.goal = None
        self.current_path = []
        self.algorithm = "A*"
        self.heuristic_type = "Manhattan"
        self.metrics = {'nodes': 0, 'cost': 0, 'time': 0}
        self.search_step_delay = 0.05
        
    def generate_maze(self, density=0.3):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.start and r == self.start.row and c == self.start.col:
                    continue
                if self.goal and r == self.goal.row and c == self.goal.col:
                    continue
                if random.random() < density:
                    self.grid[r][c].is_wall = True
                else:
                    self.grid[r][c].is_wall = False
                        
    def set_start(self, row, col):
        if self.start:
            self.start.is_start = False
        self.start = self.grid[row][col]
        self.start.is_start = True
        self.start.is_wall = False
        
    def set_goal(self, row, col):
        if self.goal:
            self.goal.is_goal = False
        self.goal = self.grid[row][col]
        self.goal.is_goal = True
        self.goal.is_wall = False
        
    def toggle_wall(self, row, col):
        node = self.grid[row][col]
        if not node.is_start and not node.is_goal:
            node.is_wall = not node.is_wall
            
    def get_neighbors(self, node):
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = node.row + dr, node.col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbor = self.grid[nr][nc]
                if not neighbor.is_wall:
                    neighbors.append(neighbor)
        return neighbors
    
    def heuristic(self, node):
        if self.heuristic_type == "Manhattan":
            return abs(node.row - self.goal.row) + abs(node.col - self.goal.col)
        else:
            return math.sqrt((node.row - self.goal.row)**2 + (node.col - self.goal.col)**2)
    
    def reset_search(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].reset()
        self.metrics = {'nodes': 0, 'cost': 0, 'time': 0}
    
    def greedy_bfs(self, gui=None):
        start_time = time.time()
        self.reset_search()
        
        pq = []
        self.start.h = self.heuristic(self.start)
        self.start.in_frontier = True
        heapq.heappush(pq, (self.start.h, id(self.start), self.start))
        
        while pq:
            _, _, current = heapq.heappop(pq)
            current.in_frontier = False
            
            if gui:
                gui.draw_grid()
                gui.draw_panel()
                pygame.display.flip()
                pygame.time.delay(int(self.search_step_delay * 1000))
            
            if current == self.goal:
                path = self.reconstruct_path(current)
                self.metrics['time'] = (time.time() - start_time) * 1000
                self.metrics['cost'] = len(path) - 1
                return path
            
            current.visited = True
            self.metrics['nodes'] += 1
            
            for neighbor in self.get_neighbors(current):
                if not neighbor.visited:
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.parent = current
                    if not neighbor.in_frontier:
                        neighbor.in_frontier = True
                        heapq.heappush(pq, (neighbor.h, id(neighbor), neighbor))
        
        self.metrics['time'] = (time.time() - start_time) * 1000
        return []
    
    def a_star(self, gui=None):
        start_time = time.time()
        self.reset_search()
        
        pq = []
        self.start.g = 0
        self.start.h = self.heuristic(self.start)
        self.start.f = self.start.g + self.start.h
        self.start.in_frontier = True
        heapq.heappush(pq, (self.start.f, id(self.start), self.start))
        
        while pq:
            _, _, current = heapq.heappop(pq)
            current.in_frontier = False
            
            if gui:
                gui.draw_grid()
                gui.draw_panel()
                pygame.display.flip()
                pygame.time.delay(int(self.search_step_delay * 1000))
            
            if current == self.goal:
                path = self.reconstruct_path(current)
                self.metrics['time'] = (time.time() - start_time) * 1000
                self.metrics['cost'] = len(path) - 1
                return path
            
            current.visited = True
            self.metrics['nodes'] += 1
            
            for neighbor in self.get_neighbors(current):
                tentative_g = current.g + 1
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.f = neighbor.g + neighbor.h
                    if not neighbor.in_frontier:
                        neighbor.in_frontier = True
                        heapq.heappush(pq, (neighbor.f, id(neighbor), neighbor))
        
        self.metrics['time'] = (time.time() - start_time) * 1000
        return []
    
    def find_path(self, gui=None):
        if self.algorithm == "A*":
            return self.a_star(gui)
        else:
            return self.greedy_bfs(gui)
    
    def reconstruct_path(self, goal_node):
        path = []
        current = goal_node
        while current:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

class Button:
    def __init__(self, text, x, y, width, height, callback):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.callback = callback
        self.is_hovered = False
        
    def draw(self, screen, font):
        color = (100, 100, 100) if self.is_hovered else (70, 70, 70)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
    def click(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.callback()
            return True
        return False

class GameGUI:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 20)
        
        self.rows = 15
        self.cols = 20
        self.cell_size = 35
        
        self.grid_width = self.cols * self.cell_size
        self.grid_height = self.rows * self.cell_size
        self.panel_width = 240
        self.window_width = self.grid_width + self.panel_width
        self.window_height = self.grid_height + 100
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Pathfinding Agent")
        
        self.agent = PathfindingAgent(self.rows, self.cols)
        self.agent.set_start(0, 0)
        self.agent.set_goal(self.rows-1, self.cols-1)
        self.agent.generate_maze(0.3)
        
        self.running = True
        self.searching = False
        self.editing_mode = "wall"
        self.create_buttons()
        
    def create_buttons(self):
        x1 = self.grid_width + 5
        x2 = self.grid_width + 85
        x3 = self.grid_width + 165
        y_start = self.grid_height + 15
        button_height = 28
        button_width = 75
        spacing = 33
        
        self.buttons = []
        
        # Row 1
        y = y_start
        self.buttons.append(Button("Find Path", x1, y, button_width, button_height, self.find_path))
        self.buttons.append(Button("A* Algo", x2, y, button_width, button_height, self.select_astar))
        self.buttons.append(Button("Manhatan", x3, y, button_width, button_height, self.select_manhattan))
        
        # Row 2
        y = y_start + spacing
        self.buttons.append(Button("Reset All", x1, y, button_width, button_height, self.reset_all))
        self.buttons.append(Button("Greedy", x2, y, button_width, button_height, self.select_greedy))
        self.buttons.append(Button("Euclidean", x3, y, button_width, button_height, self.select_euclidean))
        
        # Row 3
        y = y_start + (spacing * 2)
        self.buttons.append(Button("Random", x1, y, button_width, button_height, self.random_maze))
        self.buttons.append(Button("Clear W", x2, y, button_width, button_height, self.clear_walls))
        
    def select_astar(self):
        self.agent.algorithm = "A*"
        self.agent.current_path = []
        self.agent.reset_search()
        
    def select_greedy(self):
        self.agent.algorithm = "Greedy BFS"
        self.agent.current_path = []
        self.agent.reset_search()
        
    def select_manhattan(self):
        self.agent.heuristic_type = "Manhattan"
        self.agent.current_path = []
        self.agent.reset_search()
        
    def select_euclidean(self):
        self.agent.heuristic_type = "Euclidean"
        self.agent.current_path = []
        self.agent.reset_search()
        
    def find_path(self):
        if not self.searching:
            self.searching = True
            self.agent.current_path = self.agent.find_path(self)
            self.searching = False
        
    def reset_all(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.agent.grid[r][c].is_wall = False
                self.agent.grid[r][c].reset()
        self.agent.set_start(0, 0)
        self.agent.set_goal(self.rows-1, self.cols-1)
        self.agent.current_path = []
        self.agent.reset_search()
        
    def random_maze(self):
        self.agent.generate_maze(0.3)
        self.agent.current_path = []
        self.agent.reset_search()
        
    def clear_walls(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.agent.grid[r][c].is_start and not self.agent.grid[r][c].is_goal:
                    self.agent.grid[r][c].is_wall = False
        self.agent.current_path = []
        self.agent.reset_search()
        
    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                node = self.agent.grid[r][c]
                x = c * self.cell_size
                y = r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                if node.is_start:
                    color = (0, 255, 0)
                elif node.is_goal:
                    color = (255, 0, 0)
                elif node.is_wall:
                    color = (0, 0, 0)
                elif node.visited:
                    color = (100, 150, 255)
                elif node.in_frontier:
                    color = (255, 255, 100)
                else:
                    color = (255, 255, 255)
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
                
        if self.agent.current_path:
            for node in self.agent.current_path:
                if not node.is_start and not node.is_goal:
                    x = node.col * self.cell_size
                    y = node.row * self.cell_size
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (0, 200, 0), rect)
                    pygame.draw.rect(self.screen, (0, 100, 0), rect, 2)
                    
    def draw_panel(self):
        panel_rect = pygame.Rect(self.grid_width, 0, self.panel_width, self.grid_height + 100)
        pygame.draw.rect(self.screen, (60, 60, 60), panel_rect)
        
        current_y = 10
        
        title = self.font.render("PATHFINDING AGENT", True, (255, 255, 255))
        self.screen.blit(title, (self.grid_width + 25, current_y))
        
        current_y = 50
        pygame.draw.line(self.screen, (100, 100, 100), (self.grid_width + 10, current_y), (self.grid_width + self.panel_width - 10, current_y), 2)
        
        current_y += 15
        algo_text = self.small_font.render(f"Algorithm: {self.agent.algorithm}", True, (255, 255, 0))
        self.screen.blit(algo_text, (self.grid_width + 15, current_y))
        
        current_y += 25
        heur_text = self.small_font.render(f"Heuristic: {self.agent.heuristic_type}", True, (255, 255, 0))
        self.screen.blit(heur_text, (self.grid_width + 15, current_y))
        
        current_y += 35
        pygame.draw.line(self.screen, (100, 100, 100), (self.grid_width + 10, current_y), (self.grid_width + self.panel_width - 10, current_y), 2)
        
        current_y += 15
        nodes_text = self.small_font.render(f"Nodes Visited: {self.agent.metrics['nodes']}", True, (255, 255, 255))
        self.screen.blit(nodes_text, (self.grid_width + 15, current_y))
        
        current_y += 25
        cost_text = self.small_font.render(f"Path Cost: {self.agent.metrics['cost']}", True, (255, 255, 255))
        self.screen.blit(cost_text, (self.grid_width + 15, current_y))
        
        current_y += 25
        time_text = self.small_font.render(f"Time: {self.agent.metrics['time']:.1f} ms", True, (255, 255, 255))
        self.screen.blit(time_text, (self.grid_width + 15, current_y))
        
        current_y += 45
        pygame.draw.line(self.screen, (100, 100, 100), (self.grid_width + 10, current_y), (self.grid_width + self.panel_width - 10, current_y), 2)
        
        current_y += 15
        controls_title = self.small_font.render("CONTROLS:", True, (200, 200, 200))
        self.screen.blit(controls_title, (self.grid_width + 15, current_y))
        
        current_y += 25
        controls = [
            "Left Click: Add Wall",
            "Right Click: Remove Wall",
            "S: Set Start Point",
            "G: Set Goal Point"
        ]
        
        for text in controls:
            ctrl_text = self.small_font.render(text, True, (180, 180, 180))
            self.screen.blit(ctrl_text, (self.grid_width + 15, current_y))
            current_y += 22
        
        for button in self.buttons:
            button.draw(self.screen, self.small_font)
            
        if self.agent.algorithm == "A*":
            pygame.draw.rect(self.screen, (0, 255, 0), self.buttons[1].rect, 3)
        else:
            pygame.draw.rect(self.screen, (0, 255, 0), self.buttons[4].rect, 3)
            
        if self.agent.heuristic_type == "Manhattan":
            pygame.draw.rect(self.screen, (0, 255, 0), self.buttons[2].rect, 3)
        else:
            pygame.draw.rect(self.screen, (0, 255, 0), self.buttons[5].rect, 3)
            
    def handle_click(self, pos):
        x, y = pos
        if x < self.grid_width and y < self.grid_height:
            col = x // self.cell_size
            row = y // self.cell_size
            if 0 <= row < self.rows and 0 <= col < self.cols:
                if self.editing_mode == "start":
                    self.agent.set_start(row, col)
                    self.editing_mode = "wall"
                    self.agent.current_path = []
                    self.agent.reset_search()
                elif self.editing_mode == "goal":
                    self.agent.set_goal(row, col)
                    self.editing_mode = "wall"
                    self.agent.current_path = []
                    self.agent.reset_search()
                else:
                    if pygame.mouse.get_pressed()[0]:
                        self.agent.toggle_wall(row, col)
                        self.agent.current_path = []
                        self.agent.reset_search()
        else:
            for button in self.buttons:
                if button.click(pos):
                    break
                    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_click(pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.editing_mode = "start"
                elif event.key == pygame.K_g:
                    self.editing_mode = "goal"
                    
    def update(self):
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.update(mouse_pos)
                    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            
            self.screen.fill((200, 200, 200))
            self.draw_grid()
            self.draw_panel()
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    game = GameGUI()
    game.run()
