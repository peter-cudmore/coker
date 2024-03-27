import dataclasses

import numpy as np
from coker.toolkits.codesign import ProblemBuilder



def hexapod_codesign():
    motor_max = 0.52    # Netwon Meters

    with ProblemBuilder() as builder:
        femur_length = builder.new_variable(name='l_f')
        coxa_length = builder.new_variable(name='l_c')
        tibia_length = builder.new_variable(name='l_c')
        static_torques = builder.new_variable(r'\tau', shape=(18,))
        contact_forces = builder.new_variable('f_z', shape=(6,))
        q = builder.new_variable('q', shape=(18, ))
        q_dot = np.zeros(shape=(18,))
        q_ddot = np.zeros(shape=(18,))
        rest_height = builder.new_variable('h')

        model = build_hexapod_model(
                femur_length, coxa_length, tibia_length
        )

        constraints = [
            -0.5 * motor_max < static_torques < 0.5 * motor_max,
            -np.pi/2 < q < np.pi / 2,
            0 < femur_length < 0.2,
            0 < coxa_length < 0.2,
            0 < tibia_length < 0.2,
            rest_height > 0.2,
            0 < contact_forces
        ] + [

        ]

        # Amps... approximately
        cost = 1.47 * np.ones(18).T @ np.abs(static_torques)









