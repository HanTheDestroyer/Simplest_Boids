import pygame as pg
import numpy as np
import sys

# constants.
SCREEN_SIZE = np.array([1080, 1080])
MAX_INIT_VELOCITY = np.float64(5)
NUMBER_OF_BOIDS = 200
BOID_SIZE = 3
NEIGHBORHOOD = 40


class Simulator:
    def __init__(self):
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(np.array(SCREEN_SIZE, dtype='int16'))
        self.screen.fill(pg.Color('black'))
        self.boids = []
        for _ in np.arange(NUMBER_OF_BOIDS):
            self.boids.append(Boid())
        self.distances = []
        self.positions = []
        self.velocities = []
        self.distance_magnitudes = []

    def update(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            self.logic()
            self.screen.fill(pg.Color('black'))
            self.draw()
            pg.display.update()
            self.clock.tick(60)

    def draw(self):
        for i in np.arange(NUMBER_OF_BOIDS):
            pg.draw.circle(self.screen, color=np.array([255, 255, 255]), center=self.boids[i].pos, radius=BOID_SIZE)
            pg.draw.line(self.screen, color=np.array([255, 255, 255]),
                         start_pos=self.boids[i].pos,
                         end_pos=self.boids[i].pos - 10 * self.boids[i].vel)

    def logic(self):
        for i in np.arange(NUMBER_OF_BOIDS):
            randomness_factor = np.random.uniform(low=0.7, high=1.3, size=2)
            self.boids[i].pos[0] += self.boids[i].vel[0] * randomness_factor[0]
            self.boids[i].pos[1] += self.boids[i].vel[1] * randomness_factor[1]

            self.boids[i].pos[0] %= SCREEN_SIZE[0]
            self.boids[i].pos[1] %= SCREEN_SIZE[1]
        self.update_distances_matrix()

        for i, boid in enumerate(self.boids):
            neighbor_velocities = self.velocities[self.distance_magnitudes[i] < NEIGHBORHOOD]
            avg_neighbor_vel = np.sum(neighbor_velocities, axis=0) / len(neighbor_velocities)
            neighborhood_positions = self.positions[self.distance_magnitudes[i] < NEIGHBORHOOD]
            neighborhood_center = np.sum(neighborhood_positions, axis=0) / len(neighborhood_positions)
            distance_to_center = neighborhood_center - boid.pos
            vel_towards_center = 0
            if np.linalg.norm(distance_to_center) > NEIGHBORHOOD / 2:
                vel_towards_center = distance_to_center
            elif NEIGHBORHOOD / 4 > np.linalg.norm(distance_to_center) > 0:
                vel_towards_center = distance_to_center * -1

            boid.vel = boid.vel * 0.6 + avg_neighbor_vel * 0.4 + 0.02 * vel_towards_center

    def update_distances_matrix(self):
        position_matrix = np.zeros([NUMBER_OF_BOIDS, 2])
        velocity_matrix = np.zeros([NUMBER_OF_BOIDS, 2])
        for c in np.arange(NUMBER_OF_BOIDS):
            position_matrix[c][0] = self.boids[c].pos[0]
            position_matrix[c][1] = self.boids[c].pos[1]
            velocity_matrix[c][0] = self.boids[c].vel[0]
            velocity_matrix[c][1] = self.boids[c].vel[1]
        self.positions = position_matrix
        self.velocities = velocity_matrix
        self.distances = position_matrix[:, np.newaxis, :] - position_matrix[np.newaxis, :, :]
        self.distance_magnitudes = np.linalg.norm(self.distances, axis=2)


class Boid:
    def __init__(self, pos=None, vel=None):
        if pos is None:
            self.pos = np.array([np.random.uniform(0, SCREEN_SIZE[0]), np.random.uniform(0, SCREEN_SIZE[1])])
        else:
            self.pos = pos
        if vel is None:
            self.vel = np.array([np.random.uniform(-MAX_INIT_VELOCITY, MAX_INIT_VELOCITY),
                                 np.random.uniform(-MAX_INIT_VELOCITY, MAX_INIT_VELOCITY)])
        else:
            self.vel = vel


if __name__ == '__main__':
    simulator = Simulator()
    simulator.update_distances_matrix()
    simulator.update()
