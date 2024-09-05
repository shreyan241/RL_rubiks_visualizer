from matplotlib import pyplot as plt
from Visualizer import RubiksCubeVisualizer
from Environment import Cube

if __name__ == '__main__':
    import sys
    try:
        N = int(sys.argv[1])
        cube = Cube(N)
    except:
        N = 3
        cube = Cube(N)

    c = RubiksCubeVisualizer(cube, N)
    c.draw_interactive()
    plt.show()
