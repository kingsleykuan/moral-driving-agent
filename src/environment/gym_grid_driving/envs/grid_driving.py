from typing import List

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

import numpy as np
from collections import namedtuple
from enum import Enum

import copy

import logging
logger = logging.getLogger(__name__)

import torch
from moral_rewards.model import MoralRewardModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random = None

AgentState = Enum('AgentState', 'alive crashed finished out')

LaneSpec = namedtuple('LaneSpec', ['cars', 'speed_range'])
GridDrivingState = namedtuple('GridDrivingState',
                              ['cars', 'agent', 'finish_position', 'occupancy_trails', 'agent_state', 'observations'])
MaskSpec = namedtuple('MaskSpec', ['type', 'radius'])
FeatSpec = namedtuple('FeatSpec',
                      ['id', 'max', 'PedPed', 'Barrier', 'CrossingSignal', 'Man', 'Woman', 'Pregnant', 'Stroller',
                       'OldMan',
                       'OldWoman', 'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan',
                       'Criminal', 'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete', 'FemaleDoctor',
                       'MaleDoctor', 'Dog', 'Cat'])
ObsSpec = namedtuple('ObsSpec', ['id', 'pos'])


class DenseReward:
    FINISH_REWARD = 100
    MISSED_REWARD = -5
    CRASH_REWARD = -20
    TIMESTEP_REWARD = -1
    INVALID_CHOICE_REWARD = -100  # Crashes into invalid sides when forced to do trolley
    BARRIER_CRASH_REWARD = -10  # Crashes into barrier after observing


class SparseReward:
    FINISH_REWARD = 10
    MISSED_REWARD = 0
    CRASH_REWARD = 0
    TIMESTEP_REWARD = 0
    INVALID_CHOICE_REWARD = 0
    BARRIER_CRASH_REWARD = 0


class DefaultConfig:
    LANES = [
        LaneSpec(1, [-2, -1]),
        LaneSpec(2, [-3, -3]),
        LaneSpec(3, [-3, -1]),
    ]
    FEAT = [
        FeatSpec(1, 10, [0, 1], [0, 1], [0, 1], [0, 3], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                 [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                 [0, 10]),
        FeatSpec(2, 10, [0, 0], [0, 0], [0, 0], [0, 3], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                 [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                 [0, 10])
    ]
    OBS = [
        ObsSpec(1, (1, 1)),
        ObsSpec(2, (1, 2))
    ]

    WIDTH = 10
    PLAYER_SPEED_RANGE = [-1, -1]
    STOCHASTICITY = 1.0


class ActionNotFoundException(Exception):
    pass


class AgentCrashedException(Exception):
    pass


class AgentOutOfBoundaryException(Exception):
    pass


class AgentFinishedException(Exception):
    pass


class CarNotStartedException(Exception):
    pass


class InvalidChoiceException(Exception):
    pass


class AgentBarrierCrashException(Exception):
    pass


class AgentInObservation(Exception):
    pass


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return "Point(x={},y={})".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    @property
    def tuple(self):
        return (self.x, self.y)


class Rectangle(object):
    def __init__(self, w, h, x=0, y=0):
        self.w, self.h = w, h
        self.x, self.y = x, y

    def sample_point(self):
        return Point(random.randint(self.x, self.x + self.w), random.randint(self.y, self.y + self.h))

    def bound(self, point, bound_x=True, bound_y=True):
        x = np.minimum(np.maximum(point.x, self.x), self.x + self.w - 1) if bound_x else point.x
        y = np.minimum(np.maximum(point.y, self.y), self.y + self.h - 1) if bound_y else point.y
        return Point(x, y)

    def circular_bound(self, point, bound_x=True, bound_y=True):
        x = self.x + ((point.x - self.x) % self.w) if bound_x else point.x
        y = self.y + ((point.y - self.y) % self.h) if bound_y else point.y
        return Point(x, y)

    def contains(self, point):
        return (point.x >= self.x and point.x < self.x + self.w) and (point.y >= self.y and point.y < self.y + self.h)

    def __str__(self):
        return "Rectangle(w={},h={},x={},y={})".format(self.w, self.h, self.x, self.y)


class Observation(object):
    def __init__(self, id, pos, feat: FeatSpec):
        self.id = id
        self.pos = pos
        self.feat = feat


class Car(object):
    def __init__(self, position, speed_range, world, circular=True, auto_brake=True, auto_lane=True, p=1.0, id=None,
                 ignore=False):
        self.id = id
        self.position = position
        self.speed_range = speed_range
        self.world = world
        self.bound = self.world.boundary.circular_bound if self.world and self.world.boundary and circular else lambda \
                x, **kwargs: x
        self.auto_brake = auto_brake
        self.auto_lane = auto_lane
        self.p = p
        self.done()
        self.ignore = ignore
        self.ignored = self.ignore
        self.speed = 0

    def sample_speed(self):
        if random.random_sample() > self.p:
            return np.round(np.average(self.speed_range))
        return random.randint(*tuple(np.array(self.speed_range) + np.array([0, 1])))

    def _start(self, **kwargs):
        delta = kwargs.get('delta', Point(0, 0))
        self.destination = self.world.boundary.bound(self.bound(self.position + delta, bound_y=False), bound_x=False)
        self.changed_lane = self.destination.y != self.position.y

    def start(self, **kwargs):
        self.speed = self.sample_speed()
        self._start(delta=Point(self.speed, 0))

    def done(self):
        self.destination = None

    def _step(self, delta):
        if not self.destination:
            raise CarNotStartedException
        if not self.need_step():
            return
        if self.auto_brake and not self.can_step():
            return

        target = self.world.boundary.bound(self.bound(self.position + delta, bound_y=False), bound_x=False)

        if self.auto_lane and target.y != self.lane and not self.can_change_lane(target):
            return

        self.position = target

    def step(self, **kwargs):
        self._step(Point(self.direction if self.destination.x != self.position.x else 0,
                         np.sign(self.destination.y - self.position.y)))

    def need_step(self):
        return self.position != self.destination

    def can_step(self):
        return self.world.lanes[self.lane].gap(self)

    def can_change_lane(self, target):
        return len(filter(lambda c: c.position == target, self.world.lanes[target.y])) == 0

    @property
    def direction(self):
        return np.sign(self.speed_range[0])

    @property
    def lane(self):
        return self.position.y

    def __repr__(self):
        return "Car({}, {}, {})".format(self.id, self.position, self.speed_range)


class Action(object):
    def __init__(self, name, delta):
        self.name = name
        self.delta = delta

    def __str__(self):
        return "{}".format(self.name)

    def __repr__(self):
        return self.__str__()


class ActionableCar(Car):
    def start(self, **kwargs):
        action = kwargs.get('action')
        self._start(delta=action.delta)
        self.ignored = self.ignore or self.changed_lane


class OrderedLane(object):
    def __init__(self, world, cars=None):
        self.cars = cars or []
        self.world = world
        self.sort()

    def append(self, car):
        self.cars.append(car)
        self.sort()

    def remove(self, car):
        self.cars.remove(car)
        self.sort()

    def sort(self):
        self.cars = sorted(self.cars, key=lambda car: car.position.x, reverse=self.reverse)
        self.recognize()

    def recognize(self):
        self.cars_recognized = [car for car in self.cars if not car.ignored]

    def front(self, car):
        if car not in self.cars_recognized:  # car is ignored
            return self.cars[self.cars.index(car) - 1]
        return self.cars_recognized[self.cars_recognized.index(car) - 1]

    def gap(self, car):
        pos = [car.position.x, self.front(car).position.x][::-1 if self.reverse else 1]
        return (pos[0] - pos[1] - 1) % self.world.boundary.w

    @property
    def reverse(self):
        return False if len(self.cars) == 0 else np.sign(self.cars[0].speed_range[0]) > 0

    @property
    def ordered_cars(self):
        def first_gap_index(cars):
            for i, car in enumerate(cars):
                if self.gap(car) > 1:
                    return i

        index = first_gap_index(self.cars)
        return self.cars[index:] + self.cars[:index]

    def __repr__(self):
        view = np.chararray(self.world.boundary.w, unicode=True, itemsize=2)
        view[[car.position.x for car in filter(lambda c: isinstance(c, Car), self.cars)]] = 'O'
        view[[car.position.x for car in filter(lambda c: isinstance(c, ActionableCar), self.cars)]] = 'A'
        view[np.where(view == '')] = '-'
        return ' '.join('%03s' % i for i in view)


class Mask(object):
    def __init__(self, type, radius, boundary=None):
        assert type in ['follow', 'random']
        self.type = type
        self.radius = radius
        if self.type == 'random' and not boundary:
            raise Exception('Boundary must be defined for type: random')
        self.boundary = boundary

    def step(self, target=None):
        self.target = target
        if self.type == 'follow' and not self.target:
            raise Exception('Target must be defined for type: follow')
        if self.type == 'random':
            self.target = self.boundary.sample_point()
        return self

    def get(self):
        return Rectangle(self.radius * 2, self.radius * 2, self.target.x - self.radius, self.target.y - self.radius)

    def isInCone(self, pos):
        relative_x = self.target.x - pos.x
        relative_y = self.target.y - pos.y
        return 0 <= relative_x < self.radius and relative_x >= abs(relative_y)


class World(object):
    features: List[FeatSpec]
    observations: List[Observation]

    def __init__(self, boundary, finish_position=None, flicker_rate=0.0, mask=None):
        self.boundary = boundary
        self.finish_position = finish_position
        self.flicker_rate = flicker_rate
        self.mask = Mask(mask.type, mask.radius, self.boundary) if mask else None

    def init(self, cars, observations, features, agent=None):
        self.cars = cars
        self.ordered_cars = sorted(self.cars, key=lambda car: car.position.x + self.boundary.w * car.position.y)

        self.features = []

        for feat in features:
            people = [random.random_integers(feat.PedPed[0], feat.PedPed[1]),
                      random.random_integers(feat.Barrier[0], feat.Barrier[1]),
                      random.random_integers(feat.CrossingSignal[0], feat.CrossingSignal[1]),
                      random.random_integers(feat.Man[0], feat.Man[1]),
                      random.random_integers(feat.Woman[0], feat.Woman[1]),
                      random.random_integers(feat.Pregnant[0], feat.Pregnant[1]),
                      random.random_integers(feat.Stroller[0], feat.Stroller[1]),
                      random.random_integers(feat.OldMan[0], feat.OldMan[1]),
                      random.random_integers(feat.OldWoman[0], feat.OldWoman[1]),
                      random.random_integers(feat.Boy[0], feat.Boy[1]),
                      random.random_integers(feat.Girl[0], feat.Girl[1]),
                      random.random_integers(feat.Homeless[0], feat.Homeless[1]),
                      random.random_integers(feat.LargeWoman[0], feat.LargeWoman[1]),
                      random.random_integers(feat.LargeMan[0], feat.LargeMan[1]),
                      random.random_integers(feat.Criminal[0], feat.Criminal[1]),
                      random.random_integers(feat.MaleExecutive[0], feat.MaleExecutive[1]),
                      random.random_integers(feat.FemaleExecutive[0], feat.FemaleExecutive[1]),
                      random.random_integers(feat.FemaleAthlete[0], feat.FemaleAthlete[1]),
                      random.random_integers(feat.MaleAthlete[0], feat.MaleAthlete[1]),
                      random.random_integers(feat.FemaleDoctor[0], feat.FemaleDoctor[1]),
                      random.random_integers(feat.MaleDoctor[0], feat.MaleDoctor[1]),
                      random.random_integers(feat.Dog[0], feat.Dog[1]),
                      random.random_integers(feat.Cat[0], feat.Cat[1])]

            total = 0
            idxs = []
            while total < feat.max:
                idx = random.random_integers(0, 21)
                total += people[idx]
                idxs.append(idx)

            for i in range(0, 21):
                if not i in idxs:
                    people[i] = 0

            self.features.append(
                FeatSpec(feat.id, 0, people[0], people[1], people[2], people[3], people[4], people[5], people[6],
                         people[7], people[8], people[9], people[10], people[11], people[12], people[13], people[14],
                         people[15], people[16], people[17], people[18], people[19], people[20], people[21],
                         people[22]))

        # self.features = [
        #     FeatSpec(feat.id, 0, people[0], people[1], people[2], people[3], people[4], people[5], people[6], people[7],
        #              people[8], people[9], people[10], people[11], people[12], people[13], people[14], people[15],
        #              people[16], people[17], people[18], people[19], people[20], people[21], people[22]) for feat in
        #     features]
        self.force_decision_col = set()
        self.observations = []
        for obs in observations:
            self.force_decision_col.add(obs.pos[0])
            self.observations.append(Observation(obs.id, obs.pos, [f for f in self.features if f.id == obs.id][0]))
        self.agent = agent
        self.max_dist_travel = np.max([np.max(np.absolute(car.speed_range)) for car in cars])
        self.lanes = [OrderedLane(self) for i in range(self.boundary.h)]
        for car in cars:
            self.lanes[car.position.y].append(car)
        self.occupancy_trails = np.zeros((self.boundary.w, self.boundary.h))
        self.blackout = False
        if self.mask:
            self.mask.step(self.agent.position)
        self.agent_state = AgentState.alive
        self.update_state()

    def load(self, state):
        self.init(state.cars.union(set([state.agent])), agent=state.agent)
        self.finish_position = state.finish_position
        self.occupancy_trails = state.occupancy_trails
        self.agent_state = state.agent_state
        self.update_state()

    def reassign_lanes(self):
        unassigned_cars = []
        for y, lane in enumerate(self.lanes):
            for car in lane.cars:
                if car.position.y != y:
                    lane.remove(car)
                    unassigned_cars.append(car)
        for car in unassigned_cars:
            self.lanes[car.position.y].append(car)

    def step(self, action=None):
        exception = None
        try:
            for car in self.ordered_cars:
                if car == self.agent:
                    car.start(action=action)
                    self.lanes[car.lane].recognize()
                else:
                    car.start()

            self.occupancy_trails = np.zeros((self.boundary.w, self.boundary.h))
            for i in range(self.max_dist_travel):
                occupancies = np.zeros((self.boundary.w, self.boundary.h))
                for lane in self.lanes:
                    for car in lane.ordered_cars:
                        last_y = car.position.y

                        car.step()

                        if car != self.agent:
                            occupancies[car.position.x, car.position.y] = 1

                        if last_y != car.position.y:
                            self.reassign_lanes()

                for decision_cols in self.force_decision_col:
                    for y in range(len(self.lanes)):
                        occupancies[decision_cols, y] = 2

                for obs in self.observations:
                    if obs.feat.Barrier == 1:
                        occupancies[obs.pos] = 3
                    else:
                        occupancies[obs.pos] = 4

                # Handle car jump pass through other car
                if self.agent and occupancies[self.agent.position.x, self.agent.position.y] == 1:
                    self.agent_state = AgentState.crashed
                    raise AgentCrashedException

                # Handle car go through invalid
                if self.agent and occupancies[self.agent.position.x, self.agent.position.y] == 2:
                    self.agent_state = AgentState.crashed
                    raise InvalidChoiceException

                # Handle car go through barrier
                if self.agent and occupancies[self.agent.position.x, self.agent.position.y] == 3:
                    self.agent_state = AgentState.crashed
                    raise AgentBarrierCrashException

                # Handle car go through observation
                if self.agent and occupancies[self.agent.position.x, self.agent.position.y] == 4:
                    raise AgentInObservation

                # Handle car jump pass through finish
                if self.agent and self.finish_position and self.agent.position == self.finish_position:
                    self.agent_state = AgentState.finished
                    raise AgentFinishedException

                self.occupancy_trails = np.clip(self.occupancy_trails + occupancies, 0, 1)

            if self.agent and self.occupancy_trails[self.agent.position.x, self.agent.position.y] > 0:
                self.agent_state = AgentState.crashed
                raise AgentCrashedException
            if self.agent and not self.boundary.contains(self.agent.position):
                self.agent_state = AgentState.out
                raise AgentOutOfBoundaryException

            for car in self.ordered_cars:
                car.done()

            self.blackout = random.random_sample() <= self.flicker_rate

            if self.mask:
                self.mask.step(self.agent.position)

        except Exception as e:
            self.blackout = False
            self.mask = None
            exception = e
        finally:
            self.update_state()
            if exception:
                raise exception

    def as_tensor(self, pytorch=True):
        t = np.zeros(self.tensor_shape)
        for car in self.state.cars:
            if self.state.agent and car != self.state.agent:
                t[0, car.position.x, car.position.y] = 1
        if self.state.agent:
            t[1, self.state.agent.position.x, self.state.agent.position.y] = 1
        if self.state.finish_position:
            t[2, self.state.finish_position.x, self.state.finish_position.y] = 1
        t[3, :, :] = self.state.occupancy_trails
        for obs in self.state.observations:
            t[4, obs.pos[0], obs.pos[1]] = obs.feat.PedPed
            t[5, obs.pos[0], obs.pos[1]] = obs.feat.Barrier
            t[6, obs.pos[0], obs.pos[1]] = obs.feat.CrossingSignal
            t[7, obs.pos[0], obs.pos[1]] = obs.feat.Man
            t[8, obs.pos[0], obs.pos[1]] = obs.feat.Woman
            t[9, obs.pos[0], obs.pos[1]] = obs.feat.Pregnant
            t[10, obs.pos[0], obs.pos[1]] = obs.feat.Stroller
            t[11, obs.pos[0], obs.pos[1]] = obs.feat.OldMan
            t[12, obs.pos[0], obs.pos[1]] = obs.feat.OldWoman
            t[13, obs.pos[0], obs.pos[1]] = obs.feat.Boy
            t[14, obs.pos[0], obs.pos[1]] = obs.feat.Girl
            t[15, obs.pos[0], obs.pos[1]] = obs.feat.Homeless
            t[16, obs.pos[0], obs.pos[1]] = obs.feat.LargeWoman
            t[17, obs.pos[0], obs.pos[1]] = obs.feat.LargeMan
            t[18, obs.pos[0], obs.pos[1]] = obs.feat.Criminal
            t[19, obs.pos[0], obs.pos[1]] = obs.feat.MaleExecutive
            t[20, obs.pos[0], obs.pos[1]] = obs.feat.FemaleExecutive
            t[21, obs.pos[0], obs.pos[1]] = obs.feat.FemaleAthlete
            t[22, obs.pos[0], obs.pos[1]] = obs.feat.MaleAthlete
            t[23, obs.pos[0], obs.pos[1]] = obs.feat.FemaleDoctor
            t[24, obs.pos[0], obs.pos[1]] = obs.feat.MaleDoctor
            t[25, obs.pos[0], obs.pos[1]] = obs.feat.Dog
            t[26, obs.pos[0], obs.pos[1]] = obs.feat.Cat

        if pytorch:
            t = np.transpose(t, (0, 2, 1))  # [C, H, W]
        assert t.shape == self.tensor_space(pytorch).shape
        return t

    def as_vector(self, speed=True, trail=False):
        v = [self.state.agent.position.x, self.state.agent.position.y]
        if speed:
            v += [self.state.agent.speed]
        v += [self.state.finish_position.x, self.state.finish_position.y]
        for car in self.state.cars:
            if self.state.agent and car != self.state.agent:
                v += [car.position.x, car.position.y]
                if speed:
                    v += [car.speed]
        if trail:
            v += self.state.occupancy_trails.flatten().tolist()
        v = np.array(v)
        assert v.shape == self.vector_space(speed, trail).shape
        return v

    def tensor_space(self, pytorch=True, channel=True):
        c, w, h = self.tensor_shape
        tensor_shape = (c, h, w) if pytorch else self.tensor_shape
        return spaces.Box(low=0, high=1, shape=tensor_shape[int(not channel):], dtype=np.uint8)

    def vector_space(self, speed=True, trail=False):
        bxl, bxh = self.boundary.x, self.boundary.x + self.boundary.w
        byl, byh = self.boundary.y, self.boundary.y + self.boundary.h
        low_car_stats = [bxl, byl]
        high_car_stats = [bxh, byh]
        if speed:
            low_car_stats += [bxl]
            high_car_stats += [bxh]
        low = low_car_stats * len(self.cars) + [bxl, byl]
        high = high_car_stats * len(self.cars) + [bxh, byh]
        if trail:
            low += [0] * self.boundary.w * self.boundary.h
            high += [1] * self.boundary.w * self.boundary.h
        return spaces.Box(np.array(low), np.array(high), dtype=np.float32)

    def update_state(self):
        agent = self.agent
        other_cars = set(self.cars) - set([self.agent])
        finish_position = self.finish_position
        occupancy_trails = self.occupancy_trails
        agent_state = self.agent_state

        if self.mask:
            mask = self.mask.get()
            agent = agent if mask.contains(agent.position) else None
            other_cars = set([car for car in list(other_cars) if self.mask.isInCone(car.position)])
            finish_position = finish_position if mask.contains(finish_position) else None
            occupancy_trails_mask = np.full(self.occupancy_trails.shape, False)
            occupancy_trails_mask[mask.x:mask.x + mask.w, mask.y:mask.y + mask.h] = True
            occupancy_trails[~occupancy_trails_mask] = 0.0

        if self.blackout:
            agent = None
            other_cars = set([])
            finish_position = None
            occupancy_trails = np.zeros(self.occupancy_trails.shape)

        def cp(data):
            return copy.deepcopy(data)

        if hasattr(self, '_state'):
            del self._state
        self._state = GridDrivingState(cp(other_cars), cp(agent), cp(finish_position), cp(occupancy_trails),
                                       cp(agent_state), cp(self.observations))

    @property
    def tensor_state(self):
        return self.as_tensor()

    @property
    def vector_state(self):
        return self.as_vector()

    @property
    def state(self):
        return self._state

    @property
    def tensor_shape(self):
        return (27, self.boundary.w, self.boundary.h)


def get_range(remaining_step, current_number, target_number, step_size):
    distance = target_number - current_number

    min_range = distance - step_size * (remaining_step - 1)
    max_range = distance + step_size * (remaining_step - 1)

    min_range = max(min_range, -step_size)
    min_range = min(min_range, step_size)

    max_range = max(max_range, -step_size)
    max_range = min(max_range, step_size)

    return (min_range, max_range)


def get_trajectory(start, end, direction, step_size, boundary=None):
    assert step_size > 0 and abs(direction) in range(1, 2)
    trajectory = [start]
    x = start.x
    y = start.y
    while True:
        remaining_step = abs(end.x - x)
        if remaining_step <= 0:
            break
        a, b = get_range(remaining_step, y, end.y, step_size)
        diff = random.randint(a, b + 1)
        x += direction
        y += diff
        new = Point(x, y)
        if boundary:
            new = boundary.bound(new)
        trajectory.append(new)
    return trajectory


def sample_points_outside(points, boundary, ns):
    used_points = [] + points
    all_points = []
    for i in range(boundary.w):
        for j in range(boundary.h):
            x = i + boundary.x
            y = j + boundary.y
            all_points.append(Point(x, y))
    result = []
    for y, n in ns:
        intersection = list(set(all_points) - set(used_points))
        candidates = [p for p in intersection if p.y == y]
        points = random.choice(candidates, n).tolist()
        used_points += points
        result.append(points)
    return result


def random_speed_range(speed_range):
    min_speed = random.randint(speed_range[0], speed_range[1] + 1)
    max_speed = random.randint(min_speed, speed_range[1] + 1)
    return [min_speed, max_speed]


def generate_random_lane_speed(lanes):
    return [LaneSpec(lane.cars, random_speed_range(lane.speed_range)) for lane in lanes]


class MoralGridDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.random_seed = kwargs.get('random_seed', None)

        self.lanes = kwargs.get('lanes', DefaultConfig.LANES)
        self.width = kwargs.get('width', DefaultConfig.WIDTH)
        self.observations = kwargs.get('observations', DefaultConfig.OBS)
        self.features = kwargs.get('features', DefaultConfig.FEAT)
        self.random_lane_speed = kwargs.get('random_lane_speed', False)

        if self.random_lane_speed: self.lanes_orig = self.lanes

        self.agent_speed_range = kwargs.get('agent_speed_range', DefaultConfig.PLAYER_SPEED_RANGE)
        self.finish_position = kwargs.get('finish_position', Point(0, 0))
        self.agent_pos_init = kwargs.get('agent_pos_init', Point(self.width - 1, len(self.lanes) - 1))

        self.p = kwargs.get('stochasticity', DefaultConfig.STOCHASTICITY)
        self.observation_type = kwargs.get('observation_type', 'state')
        assert self.observation_type in ['state', 'tensor', 'vector']

        self.flicker_rate = kwargs.get('flicker_rate', 0.0)

        self.mask_spec = kwargs.get('mask', None)

        self.ensure_initial_solvable = kwargs.get('ensure_initial_solvable', False)

        self.agent_ignore = kwargs.get('agent_ignore', False)

        self.rewards = kwargs.get('rewards', SparseReward)
        rewards = [self.rewards.TIMESTEP_REWARD, self.rewards.CRASH_REWARD, self.rewards.MISSED_REWARD,
                   self.rewards.FINISH_REWARD, self.rewards.INVALID_CHOICE_REWARD, self.rewards.BARRIER_CRASH_REWARD]
        self.reward_range = (min(rewards), max(rewards))

        self.boundary = Rectangle(self.width, len(self.lanes))
        self.world = World(self.boundary, finish_position=self.finish_position, flicker_rate=self.flicker_rate,
                           mask=self.mask_spec)

        agent_direction = np.sign(self.agent_speed_range[0])
        self.actions = [Action('up', Point(agent_direction, -1)), Action('down', Point(agent_direction, 1))]
        self.actions += [Action('forward[{}]'.format(i), Point(i, 0))
                         for i in range(self.agent_speed_range[0], self.agent_speed_range[1] + 1)]

        self.action_space = spaces.Discrete(len(self.actions))

        moral_reward_model_path = kwargs.get('moral_reward_model_path', None)
        if moral_reward_model_path:
            self.moral_reward_model = MoralRewardModel.load(
                moral_reward_model_path)
            self.moral_reward_model = self.moral_reward_model.eval()
            self.moral_reward_model = self.moral_reward_model.to(device)
        else:
            self.moral_reward_model = None
            logger.warn(
                "No MoralRewardModel loaded, no moral rewards will be given.")

        self.reset()

    def seed(self, seed=None):
        global random
        random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if isinstance(action, int):
            assert action in range(len(self.actions))
            action = self.actions[action]
        assert isinstance(action, Action)

        reward = 0

        if self.done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. \
                You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            return self.state, reward, self.done, {}

        try:
            self.world.step(action)
            reward = self.rewards.TIMESTEP_REWARD
        except AgentCrashedException:
            reward = self.rewards.CRASH_REWARD
        except AgentOutOfBoundaryException:
            reward = self.rewards.MISSED_REWARD
        except AgentFinishedException:
            reward = self.rewards.FINISH_REWARD
        except InvalidChoiceException:
            reward = self.rewards.INVALID_CHOICE_REWARD
        except AgentBarrierCrashException:
            reward = self.rewards.BARRIER_CRASH_REWARD
        except AgentInObservation:
            if self.moral_reward_model is not None:
                for moral_obs in self.world.state.observations:
                    agent_x = self.world.state.agent.position.x
                    agent_y = self.world.state.agent.position.y
                    moral_obs_x = moral_obs.pos[0]
                    moral_obs_y = moral_obs.pos[1]

                    if agent_x == moral_obs_x and agent_y == moral_obs_y:
                        # Extract moral scenario features
                        moral_obs_feat = moral_obs.feat[2:]
                        moral_obs_feat = torch.tensor(
                            moral_obs_feat, dtype=torch.float, device=device)
                        moral_obs_feat = torch.unsqueeze(moral_obs_feat, 0)

                        # Get moral reward from model
                        reward = self.moral_reward_model(moral_obs_feat)
                        reward = reward['rewards'].item()
                        break
            else:
                reward = 0

        self.update_state()

        return self.state, reward, self.done, {}

    def step(self, action, state=None):
        if state is not None:
            self.load_state(state)
        return self._step(action)

    def load_state(self, state):
        if not isinstance(state, GridDrivingState):
            raise NotImplementedError

        self.world.load(state)

        self.update_observation_space()
        self.update_state()

        return self.state

    def distribute_cars(self):
        cars = []
        i = 0
        for y, lane in enumerate(self.lanes):
            choices = list(range(0, self.agent.position.x)) + list(
                range(self.agent.position.x + 1, self.width)) if self.agent.lane == y else range(self.width)
            xs = random.choice(choices, lane.cars, replace=False)
            for x in xs:
                cars.append(Car(Point(x, y), lane.speed_range, self.world, p=self.p, id=i))
                i += 1
        return cars

    def distribute_cars_solvable(self):
        cars = []
        trajectory = get_trajectory(self.agent.position, self.finish_position, self.agent.direction, 1, self.boundary)
        points = sample_points_outside(trajectory, self.boundary, [(y, lane.cars) for y, lane in enumerate(self.lanes)])
        i = 0
        for y, lane in enumerate(self.lanes):
            for point in points[y]:
                cars.append(Car(point, lane.speed_range, self.world, p=self.p, id=i))
                i += 1
        return cars

    def reset(self):
        self.seed(self.random_seed)

        if self.random_lane_speed:
            self.lanes = generate_random_lane_speed(self.lanes_orig)
            if self.random_seed is not None:
                self.random_seed += 1  # increment random seed to get different (but deterministic) lanes

        self.agent = ActionableCar(self.agent_pos_init, self.agent_speed_range, self.world, circular=False,
                                   auto_brake=False, auto_lane=False,
                                   p=self.p, id='<', ignore=self.agent_ignore)
        self.cars = [self.agent]
        if self.ensure_initial_solvable:
            self.cars += self.distribute_cars_solvable()
        else:
            self.cars += self.distribute_cars()

        self.world.init(self.cars, agent=self.agent, observations=self.observations, features=self.features)

        self.update_observation_space()

        self.update_state()

        return self.state

    def update_state(self):
        if self.observation_type == 'tensor':
            self.state = self.world.tensor_state
        elif self.observation_type == 'vector':
            self.state = self.world.vector_state
        else:
            self.state = self.world.state

    def update_observation_space(self):
        if self.observation_type == 'tensor':
            self.observation_space = self.world.tensor_space()
        elif self.observation_type == 'vector':
            self.observation_space = self.world.vector_space()
        else:
            n_cars = sum([l.cars for l in self.lanes])
            self.observation_space = spaces.Dict({
                'cars': spaces.Tuple(tuple([self.world.tensor_space(channel=False) for i in range(n_cars)])),
                'agent_pos': self.world.tensor_space(channel=False),
                'finish_pos': self.world.tensor_space(channel=False),
                'occupancy_trails': spaces.MultiBinary(self.world.tensor_space(channel=False).shape)
            })

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        cars = self.world.as_tensor(pytorch=False)[0, :, :]
        view = np.chararray(cars.shape, unicode=True, itemsize=3)
        view[self.world.state.occupancy_trails.nonzero()] = '~'
        for car in self.world.state.cars:
            if car != self.world.state.agent:
                view[car.position.tuple] = car.id or 'O'
        for col in self.world.force_decision_col:
            for y in range(len(self.lanes)):
                view[col, y] = '|'
        for obs in self.world.state.observations:
            view[obs.pos] = '@'
        if self.world.state.finish_position and self.boundary.contains(self.world.state.finish_position):
            view[self.world.state.finish_position.tuple] += 'F'
        if self.world.state.agent and self.boundary.contains(self.world.state.agent.position):
            if self.world.state.agent_state == AgentState.crashed:
                view[self.world.state.agent.position.tuple] += '#'
            else:
                view[self.world.state.agent.position.tuple] += '<'
        view[np.where(view == '')] = '-'
        view = np.transpose(view)
        print(''.join('====' for i in view[0]))
        for row in view:
            print(' '.join('%03s' % i for i in row))
        print(''.join('====' for i in view[0]))

    def close(self):
        pass

    @property
    def done(self):
        return self.world.state.agent_state != AgentState.alive
