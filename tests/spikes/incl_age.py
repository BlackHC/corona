"""
Added a copy_view context manager inspired from stackoverflow.
"""
import numpy as np
from dataclasses import dataclass

from src.copy_view import copy_view

london_age_buckets = {
    '0 <= age < 10': 1074304.0,
    '10 <= age < 20': 928524.0,
    '20 <= age < 30': 1462938.0,
    '30 <= age < 40': 1460934.0,
    '40 <= age < 50': 1166676.0,
    '50 <= age < 60': 833226.0,
    '60 <= age < 70': 599362.0,
    '70 <= age < 80': 393117.0,
    '80 <= age': 212404.0 + 42456.0
}

# From South Korea which has tested more people indiscriminately
fatality_prob_by_age_buckets = {
    '0 <= age < 10': 0,
    '10 <= age < 20': 0,
    '20 <= age < 30': 0,
    '30 <= age < 40': 0.12,
    '40 <= age < 50': 0.09,
    '50 <= age < 60': 0.37,
    '60 <= age < 70': 1.55,
    '70 <= age < 80': 5.38,
    '80 <= age': 10.22
}

assert fatality_prob_by_age_buckets.keys() == london_age_buckets.keys()

np_fatality_prob_by_age_bucket = np.array(list(fatality_prob_by_age_buckets.values())) / 100

FATALITY_RATE = 0.01
INTERACTION_COUNT = 10
INFECTION_CHANCE = 0.10

STATE_NOT_INFECTED = 0
STATE_INFECTED = 1
STATE_DEAD = 2
STATE_IMMUNE = 3

person_dtype = np.dtype(
    [('state', np.int8), ('age_bucket', np.int8), ('remaining_days', np.int8)]
)


def create_population(population_size):
    population = np.recarray(population_size, person_dtype)
    population.state = STATE_NOT_INFECTED

    age_distribution = np.array(list(london_age_buckets.values()))
    age_distribution = age_distribution / np.sum(age_distribution)

    population.age_bucket = np.random.choice(len(age_distribution), size=population_size, replace=True,
                                             p=age_distribution)
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
            post_infected_fatality_prob = np_fatality_prob_by_age_bucket[post_infected.age_bucket]
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
