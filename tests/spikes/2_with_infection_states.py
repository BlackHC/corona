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

# Main health states
PERSON_HEALTH_STATE_NOT_INFECTED = 0
PERSON_HEALTH_STATE_INFECTED = 1
PERSON_HEALTH_STATE_DEAD = 2
PERSON_HEALTH_STATE_IMMUNE = 3

# Contagion states
CONTAGION_STATE_NO = 0
CONTAGION_STATE_YES = 1
CONTAGION_STATE_NOT_ANYMORE = 2
CONTAGION_STATE_INVALID = -1

# Infection states
INFECTION_STATE_ASYMPTOMATIC = 0
INFECTION_STATE_INCUBATION = 1
INFECTION_STATE_SYMPTOMATIC = 2
INFECTION_STATE_INVALID = -1

# Remaining days: until state change
person_dtype = np.dtype(
    [('health_state', np.int8), ('age_bucket', np.int8),
     ('contagion_state', np.int8), ('contagion_remaining_days', np.int8),
     ('infection_state', np.int8), ('infection_remaining_days', np.int8)]
)


def create_population(population_size):
    population = np.recarray(population_size, person_dtype)
    population.health_state = PERSON_HEALTH_STATE_NOT_INFECTED
    population.contagion_state = CONTAGION_STATE_INVALID
    population.contagion_remaining_days = -1

    population.age_bucket = np.random.choice(
        len(np_population_distribution), size=population_size, replace=True,
        p=np_population_distribution
    )

    population.infection_state = INFECTION_STATE_INVALID
    population.infection_remaining_days = -1
    return population


def seed_infection(population, num_seeds):
    with copy_view(population, list(range(num_seeds))) as sub_population:
        infect(sub_population)


# TODO: add spike test for this? and wrap in closure to make it faster?
def draw_int_lognormal(out_median, out_sigma, size):
    mean = np.log(out_median)
    t = out_sigma ** 2 / out_median ** 2
    v = np.max(np.roots([1, -1, -t]))
    sigma = np.log(v)
    result = np.random.lognormal(mean, sigma, size).round().astype(int)
    return result


def infect(sub_population):
    assert (sub_population.health_state == PERSON_HEALTH_STATE_NOT_INFECTED).all()

    sub_population.health_state = PERSON_HEALTH_STATE_INFECTED
    sub_population.contagion_state = CONTAGION_STATE_NO

    # We expect 0.5 infections are asymptomatic
    die = 0 != np.random.binomial(1, 0.8, len(sub_population))

    with copy_view(sub_population, np.where(die)) as asymptomatic_population:
        # We use the Imperial Report here and Linton.
        asymptomatic_population.infection_state = INFECTION_STATE_ASYMPTOMATIC

        # Before we switch to `CONTAGION_STATE_YES`
        asymptomatic_population.contagion_remaining_days = draw_int_lognormal(4.6, 3.6, len(asymptomatic_population))

        # Only two states: ASYMPTOMATIC and IMMUNE
        asymptomatic_population.infection_remaining_days = asymptomatic_population.contagion_remaining_days
        # TODO: I've pulled the sigma and distribution out of my magic hat here...
        asymptomatic_population.infection_remaining_days += draw_int_lognormal(6.5, 3.6, len(asymptomatic_population))

    with copy_view(sub_population, np.where(~die)) as symptomatic_population:
        symptomatic_population.infection_state = INFECTION_STATE_INCUBATION
        # Before we switch to `INFECTION_STATE_SYMPTOMATIC`
        symptomatic_population.infection_remaining_days = draw_int_lognormal(5.1, 3.6, len(symptomatic_population))

        # We only support day steps, so we become contagious.
        # Before we switch to `CONTAGION_STATE_YES`
        symptomatic_population.contagion_remaining_days = np.fmax(symptomatic_population.infection_remaining_days - 1, 1)


def break_nonempty(sub_population):
    if len(sub_population) > 0:
        breakpoint()


def integrate(population):
    with copy_view(population, population.health_state == PERSON_HEALTH_STATE_INFECTED) as infected:
        integrate_infected(infected)

    # infect new people
    num_contagious = len(population[population.contagion_state == CONTAGION_STATE_YES])
    indices = np.random.choice(len(population), size=num_contagious * INTERACTION_COUNT, replace=True)
    unique_indices = np.unique(indices)
    infected_die = np.random.binomial(1, INFECTION_CHANCE, len(unique_indices))
    with copy_view(population, unique_indices) as candidates:
        with copy_view(candidates, np.where(infected_die)) as potential_newly_infected:
            with copy_view(potential_newly_infected,
                           potential_newly_infected.health_state == PERSON_HEALTH_STATE_NOT_INFECTED) as newly_infected:
                infect(newly_infected)


def integrate_infected(infected):
    infected.infection_remaining_days -= 1
    with copy_view(infected, infected.infection_state == INFECTION_STATE_ASYMPTOMATIC) as asymptomatic_population:
        asymptomatic_population.contagion_remaining_days -= 1

        with copy_view(asymptomatic_population, asymptomatic_population.contagion_remaining_days == 0) as mark_contagious:
            mark_contagious.contagion_state = CONTAGION_STATE_YES

        with copy_view(asymptomatic_population, asymptomatic_population.infection_remaining_days == 0) as mark_immune:
            mark_immune.infection_state = INFECTION_STATE_INVALID
            mark_immune.contagion_state = CONTAGION_STATE_INVALID
            mark_immune.health_state = PERSON_HEALTH_STATE_IMMUNE

    with copy_view(infected,
                   np.logical_and(infected.infection_state == INFECTION_STATE_INCUBATION, infected.infection_remaining_days == 0)) as mark_symptomatic:
        mark_symptomatic.infection_state = INFECTION_STATE_SYMPTOMATIC

        # TODO: how long does it take until one is healed? pulled out my hat again
        mark_symptomatic.infection_remaining_days = draw_int_lognormal(10.5, 3.6, len(mark_symptomatic))

    with copy_view(infected, np.logical_and(infected.infection_state == INFECTION_STATE_SYMPTOMATIC,
                                            infected.infection_remaining_days == 0)) as mark_immune_or_die:
        mark_immune_or_die_fatality_prob = np_fatality_by_age_prob[mark_immune_or_die.age_bucket]
        die = 0 != np.random.binomial(1, mark_immune_or_die_fatality_prob)
        mark_immune_or_die.health_state = PERSON_HEALTH_STATE_IMMUNE
        mark_immune_or_die.health_state[np.where(die)] = PERSON_HEALTH_STATE_DEAD
        mark_immune_or_die.infection_state = INFECTION_STATE_INVALID
        mark_immune_or_die.contagion_state = CONTAGION_STATE_INVALID

    with copy_view(infected, np.isin(infected.infection_state, (INFECTION_STATE_INCUBATION, INFECTION_STATE_SYMPTOMATIC))) as contagion_step:
        contagion_step.contagion_remaining_days -= 1

        with copy_view(contagion_step, contagion_step.contagion_remaining_days == 0) as update_contagion:
            # break_nonempty(update_contagion)

            with copy_view(update_contagion, update_contagion.contagion_state == CONTAGION_STATE_YES) as contagion_not_anymore:
                contagion_not_anymore.contagion_state = CONTAGION_STATE_NOT_ANYMORE

            with copy_view(update_contagion, update_contagion.contagion_state == CONTAGION_STATE_NO) as contagion_yes:
                contagion_yes.contagion_state = CONTAGION_STATE_YES
                contagion_yes.contagion_remaining_days = draw_int_lognormal(6.5, 3.6, len(contagion_yes))


@dataclass
class PopulationStats:
    alive: int
    dead: int
    infected: int
    symptomatic: int
    immune: int


def compute_stats(population):
    alive = np.sum(population.health_state != PERSON_HEALTH_STATE_DEAD).item()
    dead = np.sum(population.health_state == PERSON_HEALTH_STATE_DEAD).item()
    infected = np.sum(population.health_state == PERSON_HEALTH_STATE_INFECTED).item()
    symptomatic = np.sum(population.infection_state == INFECTION_STATE_SYMPTOMATIC).item()
    immune = np.sum(population.health_state == PERSON_HEALTH_STATE_IMMUNE).item()
    return PopulationStats(alive, dead, infected, symptomatic, immune)


population = create_population(10000000)
seed_infection(population, 3)

stats = None
for t in range(60):
    integrate(population)
    old_stats = stats
    stats = compute_stats(population)
    print(f"Day {t}:")
    if old_stats and old_stats.symptomatic > 0:
        growth_d2d = stats.symptomatic/old_stats.symptomatic - 1
        print(f"Growth d2d {growth_d2d*100}%")
    print(stats)

# print(population)
