"""
Somewhat fast and generic code for dealing with a state machine for particles.
This implements a temporal MDP essentially.
(Transition distribution and delay distribution between state changes.)

That means the state and duration until the next state are modelled.
"""
import functools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Callable, Set, Union, List

import numpy as np
from enum import Enum, EnumMeta
from toposort import toposort_flatten


# TODO: create a version of this without the Machine bit for inferred properties

class ReadOnlyContext:
    # TODO: replace this with an overloaded []?
    def get_sub_context(self, indices):
        raise NotImplementedError()


# TODO: might have to switch this to a `with ...` context :)
def get_sub_context_or(context: ReadOnlyContext, indices):
    if context is not None:
        return context.get_sub_context(indices)
    return None


class Distribution:
    def __call__(self, size, context) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class ConstantDistribution(Distribution):
    value: int

    def __call__(self, size, context) -> np.ndarray:
        return np.full(size, self.value)


@dataclass
class StateMachineSpecification:
    state_enum: EnumMeta
    states: List[int]
    start_state: int
    # A transition that will not be triggered by default but is valid.
    allowed_induced_transitions: Dict[int, Set[int]]
    transitions: Dict[int, Dict[int, Distribution]]
    transition_distributions: Dict[int, Distribution]
    state_update_order: List[int]


class StateMachineBuilder:
    states: EnumMeta
    start_state: Enum
    # A transition that will not be triggered by default but is valid.
    allowed_induced_transitions: Dict[Enum, Set[Enum]]
    transitions: Dict[Enum, Dict[Enum, Distribution]]
    transition_distributions: Dict[Enum, Distribution]

    def __init__(self, states: EnumMeta, start_state: Enum = None):
        self.states = states
        self.start_state = start_state or list(states)[0]
        self.allowed_induced_transitions = defaultdict(lambda: set())
        self.transitions = defaultdict(lambda: {})
        self.transition_distributions = {}

    def move(self, state, next_state, delay: Distribution):
        self.transitions[state][next_state] = delay

    def allow_induced_transition(self, state, next_state):
        self.allowed_induced_transitions[state].add(next_state)

    def stochastic_next_state(self, state, transition_distribution: Distribution):
        self.transition_distributions[state] = transition_distribution

    def build(self):
        int_states = [state.value for state in self.states]

        assert self.start_state in self.states
        int_start_state = self.start_state.value

        int_transitions: Dict[int, Dict[int, Distribution]] = {}
        for state, next_states in self.transitions.items():
            assert state in self.states
            int_transitions[state.value] = {}
            for next_state, delay in next_states.items():
                assert next_state in self.states
                int_transitions[state.value][next_state.value] = delay

        int_allowed_induced_transitions = {}
        for state, next_states in self.allowed_induced_transitions.items():
            assert state in self.states
            int_allowed_induced_transitions[state.value] = set()

            for next_state in next_states:
                assert next_state in self.states

                int_allowed_induced_transitions[state.value].add(next_state.value)

        int_transition_distributions = {}
        for state, distribution in self.transition_distributions.items():
            assert state in self.states
            int_transition_distributions[state.value] = distribution

        # Now build the transition order
        topo_edges: dict[int, set] = defaultdict(lambda: set())
        for state, next_states in int_transitions.items():
            for next_state in next_states.keys():
                if next_state != int_start_state:
                    topo_edges[state].add(next_state)

        # TODO: can we remove these? no need to take into account induced transitions?
        # for state, next_states in int_allowed_induced_transitions.items():
        #     for next_state in next_states:
        #         if next_state != int_start_state:
        #             topo_edges[state].add(next_state)

        toposorted = toposort_flatten(topo_edges)
        int_state_update_order = list(reversed(toposorted))

        spec = StateMachineSpecification(
            state_enum=self.states,
            states=int_states, start_state=int_start_state,
            allowed_induced_transitions=int_allowed_induced_transitions,
            transitions=int_transitions,
            transition_distributions=int_transition_distributions,
            state_update_order=int_state_update_order)
        return spec


MAX_DELAY = 128


@dataclass
class ReadOnlyStateMachineView(ReadOnlyContext):
    spec: StateMachineSpecification
    size: int
    state: np.ndarray
    next_state: np.ndarray
    delay: np.ndarray
    counter: np.ndarray

    def get_sub_context(self, indices):
        # TODO: add a size parameter (because I'm often passing a boolean mask instead of indices)
        # so its not clear whether to use count_nonzero() or len()
        return ReadOnlyStateMachineView(self.spec, self.size, self.state[indices], self.next_state[indices],
                                        self.delay[indices], self.counter[indices])


@dataclass
class StateChangeEvent:
    state: int
    next_state: int
    indices: np.ndarray


class StateMachine(ReadOnlyContext):
    spec: StateMachineSpecification
    size: int
    state: np.ndarray
    next_state: np.ndarray
    delay: np.ndarray
    counter: np.ndarray
    state_change_listeners: list

    def __init__(self, size, spec: Union[StateMachineSpecification, StateMachineBuilder]):
        if isinstance(spec, StateMachineBuilder):
            spec = spec.build()

        self.state = np.full(size, spec.start_state, 'uint8')
        self.next_state = np.full(size, spec.start_state, 'uint8')
        self.delay = np.zeros(size, 'int8')
        self.counter = np.zeros(size, 'int16')
        self.size = size
        self.spec = spec
        self.state_change_listeners = []

    def induce_transition(self, transition_mask, next_states):
        if isinstance(next_states, self.spec.state_enum):
            next_states = next_states.value
        # TODO: convert to np array and check type?

        self.delay[transition_mask] = 1
        self.next_state[transition_mask] = next_states

        # Check everything is valid.
        for state in self.spec.states:
            state_mask = np.full_like(self.state, False, dtype=np.bool)
            state_mask[transition_mask] = True
            state_mask = np.logical_and(state_mask, self.state == state)
            state_mask = np.logical_and(state_mask, self.next_state != self.state)
            num_state_particles = np.count_nonzero(state_mask)
            if num_state_particles == 0:
                continue
            # TODO: change the underlying model to be a list!!
            allowed_induced_transitions = list(self.spec.allowed_induced_transitions.get(state, set()))
            if not np.all(np.isin(self.next_state[state_mask], allowed_induced_transitions)):
                raise RuntimeError()

    def integrate(self, context=None):
        self.delay[:] -= 1
        self.counter[:] += 1

        to_update = self.delay <= 0
        self.state[to_update] = self.next_state[to_update]
        self.counter[to_update] = 0

        state_change_events = []

        for state in self.spec.state_update_order:
            state_mask = np.logical_and(to_update, self.state == state)
            num_particles = np.count_nonzero(state_mask)
            if num_particles == 0:
                continue

            possible_transitions = self.spec.transitions.get(state, {})
            if len(possible_transitions) == 0:
                self.next_state[state_mask] = state
                self.delay[state_mask] = MAX_DELAY
                state_change_events.append(StateChangeEvent(state, state, state_mask))
            elif len(possible_transitions) == 1:
                next_state, delay = list(possible_transitions.items())[0]
                self.next_state[state_mask] = next_state
                self.delay[state_mask] = delay(num_particles, get_sub_context_or(context, state_mask))

                state_change_events.append(StateChangeEvent(state, next_state, state_mask))
            else:
                # Need to figure out where we go (state-wise)
                transition_distribution = self.spec.transition_distributions.get(state, None)
                if transition_distribution is not None:
                    self.next_state[state_mask] = transition_distribution(
                        num_particles,
                        get_sub_context_or(context, state_mask))

                    # TODO: check that the distribution is valid?!?
                else:
                    raise NotImplementedError()

                for next_state, delay in possible_transitions.items():
                    next_state_mask = np.logical_and(state_mask, self.next_state == next_state)
                    num_next_state_particles = np.count_nonzero(next_state_mask)
                    if num_next_state_particles == 0:
                        continue
                    self.delay[next_state_mask] = delay(
                        num_next_state_particles,
                        get_sub_context_or(context, next_state_mask))

                    state_change_events.append(StateChangeEvent(state, next_state, next_state_mask))

        self._send_state_change_notifications(state_change_events)

    def _get_stats(self, state) -> Dict[EnumMeta, int]:
        bins = np.bincount(state)
        counter = {}
        for enum_state in self.spec.state_enum:
            if enum_state.value >= len(bins):
                counter[enum_state] = 0
            else:
                counter[enum_state] = bins[enum_state.value]
        return counter

    def get_state_stats(self):
        return self._get_stats(self.state)

    def assert_state(self, indices, state):
        if isinstance(state, self.spec.state_enum):
            state = state.value

        assert np.all(self.state[indices] == state)

    def get_sub_context(self, indices):
        view = ReadOnlyStateMachineView(
            spec=self.spec, size=self.size, state=self.state, next_state=self.next_state,
            delay=self.delay, counter=self.counter)
        return view.get_sub_context(indices)

    def _send_state_change_notifications(self, state_change_events):
        for callback in self.state_change_listeners:
            callback(state_change_events)

    def on(self, *, state: Union[Enum, Set[Enum]] = None, next_state: Union[Enum, Set[Enum]] = None):
        if state is None:
            state = set()
        if next_state is None:
            next_state = set()

        if not isinstance(state, set):
            state = {state}
        if not isinstance(next_state, set):
            next_state = {next_state}

        int_states = {state.value for state in state}
        int_next_states = {next_state.value for next_state in next_state}

        def decorator(callback):
            @functools.wraps(callback)
            def listener(state_change_events: List[StateChangeEvent]):
                for state_change_event in state_change_events:
                    if int_states and state_change_event.state not in int_states:
                        continue
                    if int_next_states and state_change_event.next_state not in int_next_states:
                        continue
                    callback(state_change_event)

            self.state_change_listeners.append(listener)
            return listener

        return decorator
