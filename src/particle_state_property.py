import functools
from enum import Enum, EnumMeta
from typing import List, Union, Set

import particle_state_machine as psm
import numpy as np
from dataclasses import dataclass


@dataclass
class StatePropertyView(psm.ReadOnlyContext):
    state_enum: EnumMeta
    size: int
    state: np.ndarray
    next_state: np.ndarray
    counter: np.ndarray

    def get_sub_context(self, indices):
        sub_state = self.state[indices]
        return StatePropertyView(state_enum=self.state_enum, size=len(sub_state),
                                 state=sub_state, next_state=self.next_state[indices],
                                 counter=self.counter[indices])


class StateProperty(psm.ReadOnlyContext):
    state_enum: EnumMeta
    size: int
    state: np.ndarray
    next_state: np.ndarray
    counter: np.ndarray
    state_change_listeners = []

    def __init__(self, start_state: Enum, size: int):
        self.size = size
        self.state_enum = type(start_state)
        self.state = np.full(size, start_state.value, 'uint8')
        self.next_state = np.full(size, start_state.value, 'uint8')
        self.counter = np.full(size, 0, 'uint16')

    def induce_transition(self, indices, next_state):
        if isinstance(next_state, self.state_enum):
            next_state = next_state.value

        self.next_state[indices] = next_state

    def integrate(self):
        self.counter += 1

        to_update = np.nonzero(self.next_state != self.state)

        state_change_events = []
        old_states, indices = np.unique(self.state[to_update], return_inverse=True)
        for i, old_state in enumerate(old_states):
            old_state_indices = to_update[indices == i]
            next_states, next_state_indices = np.unique(self.next_state[old_state_indices], return_inverse=True)
            for j, next_state in enumerate(next_states):
                indices = old_state_indices[next_state_indices == j]
                state_change_events.append(psm.StateChangeEvent(old_state, next_state, indices))

        self.state[to_update] = self.next_state[to_update]
        self.counter[to_update] = 0

        for listener in self.state_change_listeners:
            listener(state_change_events)

    # TODO: Deduplicate with StateMachine?
    def on(self, *, state: Union[Enum, Set[Enum]] = None, next_state: Union[Enum, Set[Enum]] = None):
        if state is None:
            state = {}
        if next_state is None:
            next_state = {}

        if not isinstance(state, set):
            state = {state}
        if not isinstance(next_state, set):
            next_state = {next_state}

        int_states = {state.value for state in state}
        int_next_states = {next_state.value for next_state in next_state}

        def decorator(callback):
            @functools.wraps(callback)
            def listener(state_change_events: List[psm.StateChangeEvent]):
                for state_change_event in state_change_events:
                    if int_states and state_change_event.state not in int_states:
                        return
                    if int_next_states and state_change_event.next_state not in int_next_states:
                        return
                    callback(state_change_event)

            self.state_change_listeners.append(listener)
            return listener

        return decorator

    def get_sub_context(self, indices):
        sub_context = StatePropertyView(
            state_enum=self.state_enum,
            size=self.size,
            state=self.state,
            next_state=self.next_state,
            counter=self.counter
        )
        return sub_context.get_sub_context(indices)
