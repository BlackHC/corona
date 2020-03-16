"""
Failed because numpy does not support fancy views easily.

Can I use pandas instead?
"""
import contextlib
import numpy as np

STATE_NOT_INFECTED = 0
STATE_INFECTED = 1
STATE_DEAD = 2
STATE_IMMUNE = 3

person_dtype = np.dtype(
    [('state', np.int8), ('remaining_days', np.int8)]
)


@contextlib.contextmanager
def copy_view(arr, inds):
    # create copy from fancy inds
    arr_copy = arr[inds]

    # yield 'view' (copy)
    yield arr_copy

    # after context, save modified data
    arr[inds] = arr_copy


def create_population(population_size):
    population = np.recarray(population_size, person_dtype)
    population.state = STATE_NOT_INFECTED
    population.remaining_days = 0
    return population


def seed_infection(population, num_seeds):
    with copy_view(population, list(range(num_seeds))) as sub_population:
        infect(sub_population)


def infect(sub_population):
    assert (sub_population.state == STATE_NOT_INFECTED).all()

    sub_population.state = STATE_INFECTED
    sub_population.remaining_days = np.random.randint(7, 14, len(sub_population))


FATALITY_RATE = 0.01
INTERACTION_COUNT = 10
INFECTION_CHANCE = 0.10


def integrate(population):
    with copy_view(population, population.state == STATE_INFECTED) as infected:
        infected.remaining_days -= 1

        with copy_view(infected, infected.remaining_days == 0) as post_infected:
            die = 0 != np.random.binomial(1, FATALITY_RATE, len(post_infected))
            post_infected.state = STATE_IMMUNE
            post_infected.state[np.where(die)] = STATE_DEAD

    # infect new people
    num_infected = len(population[population.state == STATE_INFECTED])
    indices = np.random.choice(len(population), size=num_infected * INTERACTION_COUNT, replace=True)
    unique_indices = np.unique(indices)
    infected_die = np.random.binomial(1, INFECTION_CHANCE, len(unique_indices))
    with copy_view(population, unique_indices) as candidates:
        with copy_view(candidates, np.where(infected_die)) as potential_newly_infected:
            with copy_view(potential_newly_infected,
                           potential_newly_infected.state == STATE_NOT_INFECTED) as newly_infected:
                infect(newly_infected)


from dataclasses import dataclass


@dataclass
class PopulationStats:
    alive: int
    dead: int
    infected: int
    immune: int


def compute_stats(population):
    alive = np.sum(population.state != STATE_DEAD).item()
    dead = np.sum(population.state == STATE_DEAD).item()
    infected = np.sum(population.state == STATE_INFECTED).item()
    immune = np.sum(population.state == STATE_IMMUNE).item()
    return PopulationStats(alive, dead, infected, immune)


population = create_population(1000000)
seed_infection(population, 3)

for t in range(48):
    integrate(population)
    print(compute_stats(population))

#print(population)