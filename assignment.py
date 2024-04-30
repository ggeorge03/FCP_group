import argparse
import random
import matplotlib.pyplot as plt
import numpy as np


class Node:
    '''Class for nodes used in a network.'''

    def __init__(self, value, number, connections=None):
        '''
        Initialises the nodes:

        - number = the index of the node
        - connections = a list of the connections between each node
        - value = value associated with the node, opinion strength.
        '''
        self.index = number
        self.connections = connections
        self.value = value


class Network:
    '''Class for a network of nodes.'''

    def __init__(self, nodes=None):
        '''
        Initialises the network:
        nodes = list of nodes in the network, default = None.
        '''
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''
        This function calculates the mean degree of a network, summing up
        all of the connections of all the nodes and dividing by the number of nodes.
        '''
        total_degree = sum(sum(node.connections) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        '''This function calculates the mean clustering coefficient of the network.'''

        total_clustering_coefficient = 0
        for node in self.nodes:
            neighbours = [self.nodes[i]
                          for i, conn in enumerate(node.connections) if conn]
            num_of_neighbours = len(neighbours)
            if num_of_neighbours < 2:
                continue
            possible_connections = num_of_neighbours * \
                (num_of_neighbours - 1) / 2
            actual_connections = 0
            for i in range(num_of_neighbours):
                for c in range(i + 1, num_of_neighbours):
                    if node.connections[neighbours[i].index] and node.connections[neighbours[c].index]:
                        actual_connections += 1
            clustering_coefficient = actual_connections / \
                possible_connections if possible_connections != 0 else 0
            total_clustering_coefficient += clustering_coefficient

        mean_clustering_coefficient = total_clustering_coefficient / \
            len(self.nodes)

        return mean_clustering_coefficient

    def get_mean_path_length(self):
        '''This function calculates the mean path length of the network.'''

        total_path_length = 0

        for node in self.nodes:
            distances = self.bfs(node)
            total_path_length += sum(distances.values())

        mean_path_length = total_path_length / \
            (len(self.nodes) * (len(self.nodes) - 1))

        return mean_path_length

    def bfs(self, start_node):
        '''This function undergoes a breadth first search from a start node.'''

        distances = {node.index: float('inf') for node in self.nodes}
        distances[start_node.index] = 0
        queue = [start_node]
        while queue:
            current_node = queue.pop()
            for neighbour_index, conn in enumerate(current_node.connections):
                if conn and distances[neighbour_index] == float('inf'):
                    distances[neighbour_index] = distances[current_node.index] + 1
                    queue.append(self.nodes[neighbour_index])
        return distances

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p:

        - N = number of nodes in the network
        - connection_probability = the probability of connections between nodes.
        '''

        self.nodes = []
        for node_number in range(N):
            # this is to generate a random float between 0 and 1 to represent the value associated with a node
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for index, node in enumerate(self.nodes):
            other_node_index = random.choice(
                [i for i in range(N) if i != index])
            node.connections[other_node_index] = 1
            self.nodes[other_node_index].connections[index] = 1

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
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

    def make_small_world_network(self, N, re_wire_prob=0.2, neighbour_range=2):
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

    def plot(self, default=True):
        '''
        This function plots figures for the ring network and the small world
        network, including using the small world network with the Ising model.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        cmap = plt.get_cmap('hot')
        plt.set_cmap(cmap)

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            if default:
                circle = plt.Circle((node_x, node_y), 1.2 * num_nodes,
                                    color=cmap(node.value))
            else:
                circle = plt.Circle((node_x, node_y), 1.2 * num_nodes,
                                    color=cmap(node.index))
            ax.add_patch(circle)

            for neighbour in node.connections:
                neighbour_index = neighbour.index
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                ax.plot((node_x, neighbour_x),
                        (node_y, neighbour_y), color='black')
        plt.show()

    def plot_network(self, N):
        '''This is the function for plotting networks using Task 3.'''

        mean_degree = self.get_mean_degree()
        mean_clustering_coefficent = self.get_mean_clustering()
        mean_path_length = self.get_mean_path_length()

        print("Mean degree:", mean_degree)
        print("Mean clustering coefficient:", mean_clustering_coefficent)
        print("Mean path length:", mean_path_length)

        coordinates_of_nodes = {node: (np.random.uniform(
            0, N), np.random.uniform(0, N)) for node in range(N)}

        plt.figure()
        for node in range(N):
            X1, Y1 = coordinates_of_nodes[node]
            plt.plot(X1, Y1, 'o', color='black')
            for neighbour_index, conn in enumerate(self.nodes[node].connections):
                if conn:
                    X2, Y2 = coordinates_of_nodes[neighbour_index]
                    plt.plot([X1, X2], [Y1, Y2], '-', color='black')
        plt.title("Random Network")
        plt.show()


def test_networks():
    '''Test function.'''
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() ==
            2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0),  network.get_mean_clustering()
    assert (network.get_mean_path_length() ==
            5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() ==
            num_nodes-1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1),  network.get_mean_clustering()
    assert (network.get_mean_path_length() ==
            1), network.get_mean_path_length()

    print("All tests passed")


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function returns the extent to which a cell agrees with its neighbours.
    It takes inputs:

    - population (numpy array), which is a grid of opinions with
                                row and col taking integer values
    - external (float), which is the strength of external opinions.

    And returns:

    - change_in_agreement (float), which is the change in agreement.
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


def calculate_node_agreement(node, external=0.0):
    '''
    This function (similar to 'calculate_agreement' function) calculates the
    change in agreement, but neighbours are now the nodes rather than perpendicular 
    cells in an array.
    '''
    current_value = node.value
    sum_agreement = 0
    for neighbour in node.connections:
        sum_agreement += neighbour.value * current_value
    change_in_node_agreement = (current_value * external) + sum_agreement
    return change_in_node_agreement


def ising_step(population, alpha=1.0, external=0.0):
    '''
    This function performs a single update of the Ising model, including opinion
    flips and external pull.
    It takes inputs:

    - population (numpy array), which is a grid of opinions with
                                row and col taking integer values
    - alpha (float), which is the tolerance parameter, controls the
                                likely-hood of opinion differences, the
                                higher the value makes flips less likely.
    - external (float), which is the strength of external opinions.
    '''
    n_rows, n_cols = population.shape  # population grid
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)

    # calc prob of flip based on disagreement
    flip_probability = min(1, np.exp(-agreement / alpha))
    if np.random.rand() < flip_probability:
        population[row, col] *= -1

    network = Network()
    for node in network.nodes:
        agreement = calculate_node_agreement(node, external)
        flip_probability = min(1, np.exp(-agreement / alpha))
        if np.random.rand() < flip_probability:
            node.value *= -1

    return population


# def ising_main_network(network, alpha=None, external=0.0, num_iterations=100):
#     '''
#     This function plots the Ising model over time for the given network.
#     Inputs:
#         network (Network): The network object on which the Ising model is to be applied.
#         alpha (float): Tolerance parameter controlling the likelihood of opinion differences.
#         external (float): Strength of external opinions.
#         num_iterations (int): Number of iterations to run the Ising model.
#     '''


def ising_with_network(N, network):
    '''
    This function runs the Ising model with networks.
    '''
    network.make_small_world_network(N, 0.2)
    # Run the Ising model on the network
    # ising_main_network(network)

    return network


def plot_ising(im, population):
    '''
    This function creating animation display of Ising model plot.
    This function is called from the ising_main function.
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows]
                      for rows in population]). astype(np.int8)  # ????
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    '''This function will test the calculate_agreement function in the Ising model.'''

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


def ising_main(population, alpha=None, external=0.0):
    '''This function plots the Ising model over time for given population.'''

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
    '''
    This is the main function where all the flags are added so that the user
    may interact with different sections of code, creating different plots
    through the command line.
    '''
    nw = Network()

    parser = argparse.ArgumentParser(
        description='Create plots for a ring network or a small world network.')
    parser.add_argument('-ring_network', type=int,
                        help='Creates ring network with inputted number of nodes')
    parser.add_argument('-small_world', type=int,
                        help='Creates small world network with inputted number of nodes')
    parser.add_argument('-re_wire', type=float, default=0.2,
                        help='Probability of rewiring for small world network')
    parser.add_argument('-range', type=float, default=2.0,
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
    parser.add_argument('-use_network', type=int, help='Uses network for the Ising model.'
                        )
    # Task 3
    parser.add_argument('-network', type=int, help='size of network')
    parser.add_argument('-test_network', action='store_true', default=False)

    args = parser.parse_args()

    if args.ring_network:
        nw.make_ring_network(
            args.ring_network, neighbour_range=args.range)
        nw.plot()

    elif args.small_world:
        nw.make_small_world_network(
            args.small_world, re_wire_prob=args.re_wire, neighbour_range=args.range)
        nw.plot()

    elif args.test_ising:
        test_ising()

    elif args.test_network:
        test_networks()

    elif args.ising_model:
        if args.use_network:
            # nw.make_small_world_network(args.use_network, 0.2)
            nw = ising_with_network(args.use_network, nw)
            nw.plot(default=False)
        else:
            population = np.random.choice([1, -1], size=(100, 100))
            ising_main(population, args.alpha, args.external)

    # elif args.use_network:
    #     ising_with_network()

    elif args.network:
        nw.make_random_network(args.network, 0.3)
        nw.plot_network(args.network)

    else:
        print('Type either -ring_network <number of nodes> or -small_world <number of nodes> to create a network.')


if __name__ == "__main__":
    main()
