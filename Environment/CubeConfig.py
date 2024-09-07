import numpy as np

# Dictionary mapping faces to color code and color acronyms
faces_to_colors = {
    'U': [0, 'W'],
    'D': [1, 'Y'],
    'L': [2, 'B'],
    'R': [3, 'G'],
    'B': [4, 'O'],
    'F': [5, 'R'],
}

# Dictionary mapping color names to their corresponding hexadecimal color codes
colors_to_hex = {
    'WHITE': '#ffffff',
    'YELLOW': '#ffd500',
    'BLUE': '#0046ad',
    'GREEN': '#009b48',
    'ORANGE': '#ff5800',
    'RED': '#cf0000',
}

# Dictionary mapping color names to their corresponding integer codes
colors_to_code = {
    'WHITE': 0,
    'YELLOW': 1,
    'BLUE': 2,
    'GREEN': 3,
    'ORANGE': 4,
    'RED': 5,
}

# Dictionary mapping each face of the cube to its adjacent faces
adj_faces = {
    0: np.array([2, 5, 3, 4]),
    1: np.array([2, 4, 3, 5]),
    2: np.array([0, 4, 1, 5]),
    3: np.array([0, 5, 1, 4]),
    4: np.array([0, 3, 1, 2]),
    5: np.array([0, 2, 1, 3])
}

move_to_rev_dict = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    10: 11,
    11: 10
}

move_dict = {
    0: ('U', 1),
    1: ('U', -1),
    2: ('D', 1),
    3: ('D', -1),
    4: ('L', 1),
    5: ('L', -1),
    6: ('R', 1),
    7: ('R', -1),
    8: ('B', 1),
    9: ('B', -1),
    10: ('F', 1),
    11: ('F', -1)
}