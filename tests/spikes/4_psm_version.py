import numpy as np
from dataclasses import dataclass

import particle_state_machine as psm
import particle_state_property as psp

from copy_view import copy_view
from enum import Enum

from data.london_population_by_age import population_by_age
from data.south_korea_fatality_by_age import fatality_by_age
from data.london_nhs_capacity import TOTAL_HOSPITAL_BEDS, TOTAL_ICU_BEDS

assert fatality_by_age.keys() == population_by_age.keys()

np_population_distribution = np.array(list(population_by_age.values()))
np_population_distribution = np_population_distribution / np.sum(np_population_distribution)

np_fatality_by_age_prob = np.array(list(fatality_by_age.values())) / 100


# TODO: add spike test for this? and wrap in closure to make it faster?
def draw_int_lognormal(out_median, out_sigma, size):
    mean = np.log(out_median)
    t = out_sigma ** 2 / out_median ** 2
    v = np.max(np.roots([1, -1, -t]))
    sigma = np.log(v)
    result = np.random.lognormal(mean, sigma, size).round().astype(int)
    return result


INTERACTION_COUNT = 10
INFECTION_CHANCE = 0.3


# Main health states
class MacroState(Enum):
    NOT_INFECTED = 0
    INFECTED = 1
    IMMUNE = 2
    DEAD = 3


# Contagion states
class ContagionState(Enum):
    INVALID = 128
    NOT_CONTAGIOUS = 0
    CONTAGIOUS = 1


# Infection states
class InfectionState(Enum):
    NONE = 128
    INCUBATION = 0
    ASYMPTOMATIC = 1
    SYMPTOMATIC = 2
    IMMUNE = 3
    DEAD = 4

def build_macro_state_spec():
    macro_state = psm.StateMachineBuilder(MacroState, MacroState.NOT_INFECTED)
    macro_state.allow_induced_transition(MacroState.NOT_INFECTED, MacroState.INFECTED)
    macro_state.allow_induced_transition(MacroState.INFECTED, MacroState.IMMUNE)
    macro_state.allow_induced_transition(MacroState.INFECTED, MacroState.DEAD)
    return macro_state.build()


macro_state_spec = build_macro_state_spec()


def build_contagion_state_spec():
    contagion_state = psm.StateMachineBuilder(ContagionState, ContagionState.INVALID)
    contagion_state.allow_induced_transition(ContagionState.INVALID, ContagionState.NOT_CONTAGIOUS)
    contagion_state.allow_induced_transition(ContagionState.CONTAGIOUS, ContagionState.INVALID)
    contagion_state.allow_induced_transition(ContagionState.NOT_CONTAGIOUS, ContagionState.INVALID)

    contagion_state.allow_induced_transition(ContagionState.NOT_CONTAGIOUS, ContagionState.CONTAGIOUS)
    contagion_state.allow_induced_transition(ContagionState.CONTAGIOUS, ContagionState.NOT_CONTAGIOUS)
    return contagion_state.build()


contagion_state_spec = build_contagion_state_spec()


def symptomatic_incubation_time(size, _):
    return draw_int_lognormal(5.1, 3.6, size)


def asymptomatic_incubation_time(size, _):
    return draw_int_lognormal(4.6, 3.6, size)


def asymptomatic_infectious_time(size, _):
    return draw_int_lognormal(4.6, 3.6, size)


def infection_incubation_transition(size, _):
    return np.random.choice([InfectionState.ASYMPTOMATIC.value, InfectionState.SYMPTOMATIC.value], size, True, p=(0.6, 0.4))


def symptomatic_death_delay(size, _):
    return draw_int_lognormal(4.6, 3.6, size)


def symptomatic_recovery_delay(size, _):
    return draw_int_lognormal(4.6, 3.6, size)


def symptomatic_outcome_transition(size, _):
    return np.random.choice([InfectionState.DEAD.value, InfectionState.IMMUNE.value], size, True, p=(0.2, 0.8))


def build_infection_state_spec():
    infection_state = psm.StateMachineBuilder(InfectionState, InfectionState.NONE)
    infection_state.allow_induced_transition(InfectionState.NONE, InfectionState.INCUBATION)

    infection_state.move(InfectionState.INCUBATION, InfectionState.SYMPTOMATIC, symptomatic_incubation_time)
    infection_state.move(InfectionState.INCUBATION, InfectionState.ASYMPTOMATIC, asymptomatic_incubation_time)
    infection_state.move(InfectionState.ASYMPTOMATIC, InfectionState.IMMUNE, asymptomatic_infectious_time)

    infection_state.stochastic_next_state(InfectionState.INCUBATION, infection_incubation_transition)

    infection_state.move(InfectionState.SYMPTOMATIC, InfectionState.DEAD, symptomatic_death_delay)
    infection_state.move(InfectionState.SYMPTOMATIC, InfectionState.IMMUNE, symptomatic_recovery_delay)

    infection_state.stochastic_next_state(InfectionState.SYMPTOMATIC, symptomatic_outcome_transition)

    return infection_state.build()


infection_state_spec = build_infection_state_spec()


@dataclass
class World(psm.ReadOnlyContext):
    age_bucket: np.ndarray
    macro_state: psm.StateMachine
    infection_state: psm.StateMachine
    contagion_state: psm.StateMachine

    def get_sub_context(self, indices):
        return World(
            age_bucket=self.age_bucket[indices], macro_state=self.macro_state.get_sub_context(indices),
            infection_state=self.infection_state.get_sub_context(indices),
            contagion_state=self.contagion_state.get_sub_context(indices))

    def seed_infection(self, num_seeds):
        self.infect(list(range(num_seeds)))

    @staticmethod
    def create(population_size):
        age_bucket = np.random.choice(
            len(np_population_distribution), size=population_size, replace=True,
            p=np_population_distribution
        )
        macro_state = psm.StateMachine(population_size, macro_state_spec)
        infection_state = psm.StateMachine(population_size, infection_state_spec)
        contagion_state = psm.StateMachine(population_size, contagion_state_spec)

        @macro_state.on(state=MacroState.DEAD)
        def reset_other_states(state_change_event: psm.StateChangeEvent):
            infection_state.induce_transition(state_change_event.indices, InfectionState.NONE)
            contagion_state.induce_transition(state_change_event.indices, ContagionState.INVALID)

        @infection_state.on(state=InfectionState.DEAD)
        def propagate_death(state_change_event: psm.StateChangeEvent):
            macro_state.induce_transition(state_change_event.indices, MacroState.DEAD)
            contagion_state.induce_transition(state_change_event.indices, ContagionState.NOT_CONTAGIOUS)

        @infection_state.on(state=InfectionState.IMMUNE)
        def propagate_immunity(state_change_event: psm.StateChangeEvent):
            macro_state.induce_transition(state_change_event.indices, MacroState.IMMUNE)
            contagion_state.induce_transition(state_change_event.indices, ContagionState.NOT_CONTAGIOUS)

        return World(age_bucket, macro_state, infection_state, contagion_state)

    def infect(self, indices):
        self.macro_state.assert_state(indices, MacroState.NOT_INFECTED)

        self.macro_state.induce_transition(indices, MacroState.INFECTED)
        self.contagion_state.induce_transition(indices, ContagionState.NOT_CONTAGIOUS)
        self.infection_state.induce_transition(indices, InfectionState.INCUBATION)

    def integrate(self):
        self.infection_state.integrate()
        self.contagion_state.integrate()

        self.macro_state.integrate()

        to_update = np.logical_and(self.infection_state.state == InfectionState.INCUBATION.value, self.infection_state.delay <= 1)
        self.contagion_state.induce_transition(to_update, ContagionState.CONTAGIOUS)

        # infect new people
        contagious_people = self.contagion_state.state == ContagionState.CONTAGIOUS.value
        num_contagious = np.count_nonzero(contagious_people)
        indices = np.random.choice(self.macro_state.size, size=num_contagious * INTERACTION_COUNT, replace=True)
        unique_indices = np.unique(indices)
        infected_die = np.random.binomial(1, INFECTION_CHANCE, len(unique_indices))
        potential_infected_indices = unique_indices[infected_die != 0]
        newly_infected_indices = potential_infected_indices[self.macro_state.state[potential_infected_indices] == MacroState.NOT_INFECTED.value]
        self.infect(newly_infected_indices)


world = World.create(1000000)
world.seed_infection(30)

stats = None
for t in range(60):
    world.integrate()
    stats = world.macro_state.get_state_stats()
    # old_stats = stats
    # stats = compute_stats(population)
    # print(f"Day {t}:")
    # if old_stats and old_stats.symptomatic > 0:
    #     growth_d2d = stats.symptomatic / old_stats.symptomatic - 1
    #     print(f"Growth d2d {growth_d2d * 100}%")
    print(stats)
    #more_stats = world.infection_state.get_state_stats()
    #print(more_stats)
    #print(world.contagion_state.get_state_stats())

# print(population)
