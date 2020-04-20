import particle_state_machine
from enum import Enum
import numpy as np


def test_builder():
    class State(Enum):
        START = 0
        FIRST = 1
        SECOND = 2
        END = 3

    builder = particle_state_machine.StateMachineBuilder(State)
    builder.move(State.FIRST, State.SECOND, particle_state_machine.ConstantDistribution(1))
    builder.move(State.SECOND, State.END, particle_state_machine.ConstantDistribution(1))
    builder.allow_induced_transition(State.START, State.FIRST)

    psm = particle_state_machine.StateMachine(10, builder)
    psm.integrate()
    assert psm.get_state_stats() == {State.START: 10, State.FIRST: 0, State.SECOND: 0, State.END: 0}
    psm.induce_transition(np.array(True), State.FIRST.value)
    assert psm.get_state_stats() == {State.START: 10, State.FIRST: 0, State.SECOND: 0, State.END: 0}
    psm.integrate()
    assert psm.get_state_stats() == {State.START: 0, State.FIRST: 10, State.SECOND: 0, State.END: 0}
    psm.integrate()
    assert psm.get_state_stats() == {State.START: 0, State.FIRST: 0, State.SECOND: 10, State.END: 0}
    psm.integrate()
    assert psm.get_state_stats() == {State.START: 0, State.FIRST: 0, State.SECOND: 0, State.END: 10}
