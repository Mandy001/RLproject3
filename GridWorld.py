import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # pixels


class GridWorld(tk.Tk, object):
    def __init__(self, grid_world_h=5, grid_world_w=5):
        super(GridWorld, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)

        self.grid_world_h = grid_world_h
        self.grid_world_w = grid_world_w

        self.title('GridWorld')
        self.geometry('{0}x{1}'.format((grid_world_w + 1) * UNIT, (grid_world_h + 1) * UNIT))
        self._build_grid_world()

    def _build_grid_world(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=(self.grid_world_h + 1) * UNIT,
                                width=(self.grid_world_w + 1) * UNIT)

        # draw column line
        for c in range(0, self.grid_world_w * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.grid_world_h * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        # draw row line
        for r in range(0, self.grid_world_h * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.grid_world_w * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # draw x index
        for c in range(20, self.grid_world_w * UNIT, UNIT):
            x, y = c, self.grid_world_h * UNIT + 20
            self.canvas.create_text(x, y, text=str((c - 20) // UNIT))

        for r in range(20, self.grid_world_h * UNIT, UNIT):
            x, y = self.grid_world_w * UNIT + 20, r
            self.canvas.create_text(x, y, text=str((r - 20) // UNIT))

        # create origin
        origin = np.array([20, 20])

        self.wall_coords_list = self.build_wall([(2, 2), (2, 3)])

        target_w, target_h = (3, 3)
        target_center = origin + np.array([UNIT * target_w, UNIT * target_h])
        self.target = self.canvas.create_oval(
            target_center[0] - 15, target_center[1] - 15,
            target_center[0] + 15, target_center[1] + 15,
            fill='yellow')

        # create red rect
        start_w, start_h = (0, 0)
        current_center = origin + np.array([UNIT * start_w, UNIT * start_h])
        self.current = self.canvas.create_rectangle(
            current_center[0] - 15, current_center[1] - 15,
            current_center[0] + 15, current_center[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def build_wall(self, wall_list):
        wall_coords_list = []

        origin = np.array([20, 20])
        for wall in wall_list:
            w, h = wall
            wall_center = origin + np.array([UNIT * w, UNIT * h])
            wall = self.canvas.create_rectangle(
                wall_center[0] - 15, wall_center[1] - 15,
                wall_center[0] + 15, wall_center[1] + 15,
                fill='black')
            wall_coords = self.canvas.coords(wall)
            wall_coords_list.append(wall_coords)
        return wall_coords_list

    def reset(self):
        self.update()
        # time.sleep(0.5)
        self.canvas.delete(self.current)
        origin = np.array([20, 20])
        self.current = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        coords = self.canvas.coords(self.current)
        idx = self.convert_coords_to_index(coords)
        return coords, idx

    def step(self, action):
        s = self.canvas.coords(self.current)

        # only for DP
        if s == self.canvas.coords(self.target):
            reward = 20
            done = True
            next_state_coords = 'terminal'
            next_state_index = 'terminal'
            return next_state_coords, next_state_index, reward, done

        base_action = np.array([0, 0])
        current_center = np.array([s[0] + 15, s[1] + 15])

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (self.grid_world_h - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if s[0] < (self.grid_world_w - 1) * UNIT:
                base_action[0] += UNIT

        next_center = current_center + base_action
        next_state_coords = [next_center[0] - 15, next_center[1] - 15, next_center[0] + 15, next_center[1] + 15]

        # reward function
        # find target
        if next_state_coords == self.canvas.coords(self.target):
            self.canvas.move(self.current, base_action[0], base_action[1])
            reward = 1
            done = True
            next_state_coords = 'terminal'
            next_state_index = np.array([0, 0])
        # touch the wall
        elif next_state_coords in self.wall_coords_list:
            reward = 0
            done = False
            next_state_coords = s
            next_state_index = self.convert_coords_to_index(s)
        # normal step
        else:
            self.canvas.move(self.current, base_action[0], base_action[1])
            next_state_index = self.convert_coords_to_index(next_state_coords)
            reward = 0
            done = False

        return next_state_coords, next_state_index, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()

    def set_current_state(self, x_index, y_index):
        self.update()
        # time.sleep(0.5)
        self.canvas.delete(self.current)
        origin = np.array([20 + x_index * 40, 20 + y_index * 40])
        self.current = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        coords = self.canvas.coords(self.current)
        idx = self.convert_coords_to_index(coords)
        return coords, idx

    def convert_coords_to_index(self, coords):
        x_0, y_0, x_1, y_1 = coords
        x_center = (x_0 + 15 - 20) / UNIT
        y_center = (y_0 + 15 - 20) / UNIT
        return np.array([int(x_center), int(y_center)])


def update():
    for t in range(10):
        s, _ = env.reset()
        while True:
            env.render()
            a = 1
            s, _, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = GridWorld()
    env.after(100, update)
    env.mainloop()
