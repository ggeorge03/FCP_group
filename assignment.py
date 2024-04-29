import argparse
import random
import matplotlib.pyplot as plt
import numpy as np


class Node:
    '''Class for nodes.'''

    def __init__(self, value, number, connections=None):
        '''Initialises the nodes'''
        self.index = number
        self.connections = connections
        self.value = value


class Network:
    '''Class for network'''

    def __init__(self, nodes=None):
        '''Initialises the network.'''
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''Add docstring.'''
        total_degree = sum(sum(node.connections) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        '''Add docstring.'''
        total_clustering_coefficient = 0
        for node in self.nodes:
            neighbors = [self.nodes[i]
                         for i, conn in enumerate(node.connections) if conn]
            num_neighbors = len(neighbors)
            possible_connections = num_neighbors * (num_neighbors - 1) / 2
            actual_connections = sum(
                node.connections[nei.index] for nei in neighbors)
            clustering_coefficient = actual_connections / \
                possible_connections if possible_connections != 0 else 0
            total_clustering_coefficient += clustering_coefficient

        mean_clustering_coefficient = total_clustering_coefficient / \
            len(self.nodes)

        return mean_clustering_coefficient

    def get_mean_path_length(self):
        '''Add docstring.'''
        total_path_length = 0

        for node in self.nodes:
            distances = self.bfs(node)
            total_path_length += sum(distances.values())

        return total_path_length / (len(self.nodes) * (len(self.nodes) - 1))

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p.
        '''
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        '''
        This function creates a ring network where each node is connected
        to all nodes within a specificed range.

        N = number of nodes in the network.

        neighbour_range = within this range, all nodes are connected.
        Default is 1, meaning each node is connected to its immediate
        neighbours only.
        '''
        # Initialises the nodes of the network in a list.
        self.nodes = [Node(i / N, i) for i in range(N)]

        # Establishes the connections between each node making
        # sure the connections wrap around the network.
        for i, node in enumerate(self.nodes):
            node.connections = [self.nodes[(i + j) % N]
                                for j in range(1, int(neighbour_range + 1))]

    def make_small_world_network(self, N, re_wire_prob=0.2, neighbour_range=1):
        '''
        This function creates a small world network that starts with a ring
        network and then rewires edges.

        N = number of nodes in the network.

        re_wire_prob = the probability of rewiring an edge. Deafult is 0.2.
        '''
        self.make_ring_network(
            N, neighbour_range)  # Start with a ring network.

        # For each node, each edge is rewired with a given
        # probability to a new randomly selected node.
        # It disallows self-connections and repeat connections.
        for node in self.nodes:
            for i, neighbour in enumerate(node.connections):
                if random.random() < re_wire_prob:
                    new_neighbour = random.choice(self.nodes)
                    while new_neighbour == node or new_neighbour in node.connections:
                        new_neighbour = random.choice(self.nodes)
                    # Updates and applies the new connection.
                    node.connections[i] = new_neighbour

    def plot(self):
        '''Plot figures.'''

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        cmap = plt.get_cmap('hot')
        # node_degrees = self.node_degree()
        # max_degree = max(node_degrees)
        plt.set_cmap(cmap)

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            # normalized_degree = node_degrees[i] / max(node_degrees)

            # node_colour = cmap(normalized_degree)

            # node_degree = node_degrees[i]
            # circle_colour = cmap(node_degree / max_degree)

            # circle = plt.Circle((node_x, node_y), 1.2 * num_nodes,
            #                     color=circle_colour)

            circle = plt.Circle((node_x, node_y), 1.2 * num_nodes,
                                color=cmap(node.value))
            ax.add_patch(circle)

            for neighbour in node.connections:
                neighbour_index = neighbour.index
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                ax.plot((node_x, neighbour_x),
                        (node_y, neighbour_y), color='black')
        plt.show()


# function to calculate neighbour agreement
def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs:
        population (numpy array) - grid of opinions
        row (int)
        col (int)
        external (float) - strength of external opinion
    Returns:
        change_in_agreement (float) - change in agreement
    '''
    current_value = population[row, col]  # gets initial cell value
    n_rows, n_cols = population.shape
    sum_agreement = 0
    neighbours = [
        ((row - 1) % n_rows, col),
        ((row + 1) % n_rows, col),
        (row, (col - 1) % n_cols),
        (row, (col + 1) % n_cols)
    ]  # define neighbours(above, below, right, left)
    for r, c in neighbours:
        sum_agreement += population[r, c] * current_value
    # equation to calulate change in agreement
    change_in_agreement = (current_value * external) + sum_agreement
    return change_in_agreement


# function to update Ising model to include changing opinions and external factors.
def ising_step(population, alpha=1.0, external=0.0):
    '''
    This function performs a single update of the Ising model, including opinion flips and external pull.
    Inputs:
        population (numpy array) - grid of opinions
        alpha (float) - tolerance parameter, controls the likely-hood of opinion differences, the higher the value makes flips less likely.
        external (float) - strength of external opinions
    '''
    n_rows, n_cols = population.shape  # population grid
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)
    # calc prob of flip based on disagreement
    flip_probability = min(1, np.exp(-agreement / alpha))
    if np.random.rand() < flip_probability:
        population[row, col] *= -1
    return population


def plot_ising(im, population):  # function to display model.
    '''
    This function displays a plot of the Ising model.
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows]
                      for rows in population]). astype(np.int8)  # ????
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():  # function to test model.
    '''
    This function will test the calculate_agreement function in the Ising model
    '''
    print("Testing Ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    # Testing external pull
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):  # Main function
    '''
    This function plots the Ising model over time for given population.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    # Iterating an update 100 times
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
    for frame in range(100):
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


def main():
    '''g'''
    network = Network()

    parser = argparse.ArgumentParser(
        description='Create plots for a ring network or a small world network.')
    parser.add_argument('-ring_network', type=int,
                        help='Creates ring network with inputted number of nodes')
    parser.add_argument('-small_world', type=int,
                        help='Creates small world network with inputted number of nodes')
    parser.add_argument('-re_wire', type=float, default=0.2,
                        help='Probability of rewiring for small world network')
    parser.add_argument('-range', type=float, default=1.0,
                        help='Neighbour range')
    # Ising model
    parser.add_argument('-ising_model', action='store_true',
                        help='Run Ising Model')
    parser.add_argument('-test_ising', action='store_true',
                        help='Test Ising Model')
    parser.add_argument('-external', type=float, default=0.0,
                        help='Strength of external opinion')
    parser.add_argument('-alpha', type=float, default=1.0,
                        help='Tolerance parameter for opinion differences.')

    args = parser.parse_args()

    if args.ring_network:
        network.make_ring_network(
            args.ring_network, neighbour_range=args.range)
        network.plot()

    elif args.small_world:
        network.make_small_world_network(
            args.small_world, re_wire_prob=args.re_wire, neighbour_range=args.range)
        network.plot()

    elif args.test_ising:
        test_ising()

    elif args.ising_model:
        population = np.random.choice([1, -1], size=(100, 100))
        ising_main(population, args.alpha, args.external)

    else:
        print('Type either -ring_network <number of nodes> or -small_world <number of nodes> to create a network.')


if __name__ == "__main__":
    main()
