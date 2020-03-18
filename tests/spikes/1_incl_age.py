"""
Added a copy_view context manager inspired from stackoverflow.
"""
import numpy as np
from dataclasses import dataclass

from src.copy_view import copy_view

from data.london_population_by_age import population_by_age
from data.south_korea_fatality_by_age import fatality_by_age

assert fatality_by_age.keys() == population_by_age.keys()

np_population_distribution = np.array(list(population_by_age.values()))
np_population_distribution = np_population_distribution / np.sum(np_population_distribution)

np_fatality_by_age_prob = np.array(list(fatality_by_age.values())) / 100

INTERACTION_COUNT = 10
INFECTION_CHANCE = 0.10

STATE_NOT_INFECTED = 0
STATE_INFECTED = 1
STATE_DEAD = 2
STATE_IMMUNE = 3


# Remaining days: until state change
person_dtype = np.dtype(
    [('state', np.int8), ('age_bucket', np.int8), ('remaining_days', np.int8)]
)


def create_population(population_size):
    population = np.recarray(population_size, person_dtype)
    population.state = STATE_NOT_INFECTED

    population.age_bucket = np.random.choice(
        len(np_population_distribution), size=population_size, replace=True,
        p=np_population_distribution
    )

    population.remaining_days = 0
    return population


def seed_infection(population, num_seeds):
    with copy_view(population, list(range(num_seeds))) as sub_population:
        infect(sub_population)


def infect(sub_population):
    assert (sub_population.state == STATE_NOT_INFECTED).all()

    sub_population.state = STATE_INFECTED
    sub_population.remaining_days = np.random.randint(7, 14, len(sub_population))


def integrate(population):
    with copy_view(population, population.state == STATE_INFECTED) as infected:
        infected.remaining_days -= 1

        with copy_view(infected, infected.remaining_days == 0) as post_infected:
            post_infected_fatality_prob = np_fatality_by_age_prob[post_infected.age_bucket]
            die = 0 != np.random.binomial(1, post_infected_fatality_prob)
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


population = create_population(10000)
seed_infection(population, 3)

for t in range(48):
    integrate(population)
    print(compute_stats(population))

# print(population)
