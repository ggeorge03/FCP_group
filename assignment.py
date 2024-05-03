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
        - nodes = list of nodes in the network, default = None.
        '''
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_neighbours(self, node_index):
        '''
        This function returns the indices of neighbouring nodes
        connected to the node with the specified index.
        '''
        node = self.nodes[node_index]  # Finds the indices of connected nodes.
        neighbours = [i for i, conn in enumerate(
            node.connections) if conn == 1]
        return neighbours

    def get_mean_degree(self):
        '''
        This function calculates the mean degree of a network, summing up
        all of the connections of all the nodes and dividing by the number of nodes.
        '''
        total_degree = sum(sum(node.connections) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        '''This function calculates the mean clustering coefficient of the network.'''
        total_clustering_coefficient = 0  # Create total clustering coefficient
        # Iterates through each node in network
        for node_index in range(len(self.nodes)):
            # Gets neighbours for each node
            neighbours = self.get_neighbours(node_index)
            num_of_neighbours = len(neighbours)  # Store number of neighbours
            # Calculates total number of possible connections between neighbours.
            possible_connections = num_of_neighbours * \
                (num_of_neighbours - 1) / 2
            actual_connections = 0  # initialise count for actual connections
            # Iterates through pairs of neighbours to count actual connections
            for i in range(num_of_neighbours):
                for c in range(i + 1, num_of_neighbours):
                    # If two neighbours are connected, increase the count
                    if self.nodes[neighbours[i]].connections[neighbours[c]]:
                        actual_connections += 1
            # Calculates clustering coefficient for the node
            clustering_coefficient = actual_connections / \
                possible_connections if possible_connections != 0 else 0
            total_clustering_coefficient += clustering_coefficient  # Adds it to total
        # Calculates the mean clustering coefficient for the network
        mean_clustering_coefficient = total_clustering_coefficient / \
            len(self.nodes)
        return mean_clustering_coefficient

    def get_mean_path_length(self):
        '''Calculates the mean path length of the network.'''
        mean_paths = []  # Create list to store mean path length of each node
        # Iterates through each node in network
        for node in self.nodes:
            # Apply a breadth first search that calculates path lengths from the current node to all other nodes
            path_lengths = self.bfs(node.index)
            mean_paths.append(np.mean(path_lengths[path_lengths > 0]))
        mean_path_length = np.mean(mean_paths)
        return round(mean_path_length, 15)

    def bfs(self, start_node):
        '''This function undergoes a breadth first search from a start node.'''
        num_nodes = len(self.nodes)  # Defines number of nodes in network

        # initialises arrays that store path lengths and visited nodes
        path_lengths = np.zeros(num_nodes, dtype=int)
        visited = np.zeros(num_nodes, dtype=bool)

        # initialises queue beginning with start node
        unvisited = [start_node]
        visited[start_node] = True  # saved start node as visited
        n = 0  # path length counter

        # start breadth first search traversal
        while unvisited:
            next_unvisited = []  # list to store nodes to visit next
            for current_node_index in unvisited:
                # iterates over all neighbours of the current node
                for neighbour_index in self.get_neighbours(current_node_index):
                    # check if neighbour has been NOT visited
                    if not visited[neighbour_index]:
                        path_lengths[neighbour_index] = n + 1
                        next_unvisited.append(neighbour_index)
                        # marks neighbour as visited
                        visited[neighbour_index] = True
            unvisited = next_unvisited  # updates list of unvisited
            n += 1  # increments path length counter by 1
        return path_lengths

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p:
        - N = number of nodes in the network
        - connection_probability = the probability of connections between nodes.
        '''
        self.nodes = []  # list to store nodes in the network
        # generates random float between 0 and 1 representing the value associated with a node
        for node_number in range(N):
            value = np.random.random()
            # Initialise node connections as zero
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        # Creates connections between nodes (task3)
        for index, node in enumerate(self.nodes):
            # Selects random node to connect to
            other_node_index = random.choice(
                [i for i in range(N) if i != index])
            node.connections[other_node_index] = 1
            self.nodes[other_node_index].connections[index] = 1

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                # checks if connections should be made based on the connection probability
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        '''
        This function creates a ring network where each node is connected
        to all nodes within a specificed range.
        - N = number of nodes in the network.
        - neighbour_range = within this range, all nodes are connected.
        - Default = 1, meaning each node is connected to its immediate neighbours only.
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
        - N = Number of nodes in the network.
        - re_wire_prob = Probability of rewiring an edge.
        - Default = 0.2.
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
        fig = plt.figure('Small World Network', figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
        cmap = plt.get_cmap('hot')  # Choose colourmap 'hot'
        plt.set_cmap(cmap)

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            if default:
                circle = plt.Circle((node_x, node_y), 1.2 *
                                    num_nodes, color=cmap(node.value))
            else:
                circle = plt.Circle((node_x, node_y), 1.2 *
                                    num_nodes, color=cmap(node.value))
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

        # Prints the calculated network figures
        print("Mean degree:", mean_degree)
        print("Mean clustering coefficient:", mean_clustering_coefficent)
        print("Mean path length:", mean_path_length)

        # Generates random coordinates for each node
        coordinates_of_nodes = {node: (np.random.uniform(
            0, N), np.random.uniform(0, N)) for node in range(N)}
        plt.figure('Task 3: Networks', figsize=(10, 8))

        for node in range(N):
            # Gets coords for current node and plots it
            X1, Y1 = coordinates_of_nodes[node]
            plt.plot(X1, Y1, 'o', color='black')
            # Iterates over connections of current node
            for neighbour_index, conn in enumerate(self.nodes[node].connections):
                if conn:
                    X2, Y2 = coordinates_of_nodes[neighbour_index]
                    # Plots line to connect two nodes that are neighbours
                    plt.plot([X1, X2], [Y1, Y2], '-', color='black')
        # Set variable plot title using fstring based on user input
        plt.title(f'Random Network with {N} nodes')
        plt.show()


def test_networks():
    '''Test function.'''  # Ring network
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
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
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
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() ==
            1), network.get_mean_path_length()
    print("All tests passed")


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function returns the extent to which a cell agrees with its neighbours.
    Inputs:
        - population (numpy array) = Grid of opinions with row and col taking integer values.
        - external (float) = Strength of external opinions.
    Returns:
        - change_in_agreement (float) = Change in agreement.
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
        # equation to calulate change in agreement
        sum_agreement += population[r, c] * current_value
    change_in_agreement = (current_value * external) + sum_agreement
    return change_in_agreement


def calculate_node_agreement(node, external=0.0):
    '''
    This function (similar to 'calculate_agreement' function) calculates the
    change in agreement, but neighbours are now the nodes rather than perpendicular
    cells in an array.
    '''
    current_value = node.value  # Gets current value of node (strength of opinion)
    sum_agreement = 0

    for neighbour in node.connections:
        sum_agreement += neighbour.value * current_value
    # Change in agreement is calculated based on external influence
    change_in_node_agreement = (current_value * external) + sum_agreement
    return change_in_node_agreement


def ising_step(population, alpha=1.0, external=0.0):
    '''
    This function performs a single update of the Ising model, including opinion
    flips and external pull.
    Inputs:
        - population (numpy array) = Grid of opinions with row and col taking integer values.
        - alpha (float) = Tolerance parameter, controls the likely-hood of opinion differences, the higher the value makes flips less likely.
        - external (float) = Strength of external opinions.
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


def ising_step_network(network, alpha=1.0, external=0.0):
    '''
    This function runs the Ising model on the given network.
    Inputs:
        - network (Network) = The network object on which the Ising model is to be applied.
        - alpha (float) = Tolerance parameter controlling the likelihood of opinion differences.
        - external (float) = Strength of external opinions.
        - num_iterations (int) = Number of iterations to run the Ising model.
    '''
    # Iterates over all nodes in network
    for node in network.nodes:
        # Calculates agreement for current node
        agreement = calculate_node_agreement(node, external)
        # Probability of flipping opinion
        flip_probability = min(1, np.exp(-agreement / alpha))
        # Flip opinion if < probabilty
        if np.random.rand() < flip_probability:
            node.value *= -1


def plot_ising(im, population):
    '''
    This function creates the animation display of the Ising model plot.
    This function is called from the ising_main function.
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows]
                      for rows in population]). astype(np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def ising_main(population, alpha=None, external=0.0):
    '''This function plots the Ising model over time for a given population.'''
    fig = plt.figure('Ising Model', figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plt.title(f'External: {external}, Alpha: {alpha}')
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    for frame in range(100):  # Iterating an update 100 times
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


def ising_with_network(N, network):
    '''This function runs the Ising model with networks.'''
    fig = plt.figure('Ising Model with Network', figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    network.make_small_world_network(N, 0.2)

    num_nodes = len(network.nodes)
    if num_nodes == 0:
        return network

    network_radius = num_nodes * 10
    ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
    ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
    cmap = plt.get_cmap('hot')
    im = None  # Initialize im to None

    for frame in range(100):  # Iterating an update 100 times
        for step in range(1000):
            network.make_small_world_network(N, 0.2)
            ising_step_network(network)
        print('Step:', frame, end='\r')

        if im:
            im.remove()  # Remove the previous plot
        # Plot nodes
        for node in network.nodes:
            node_angle = node.index * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)
            circle = plt.Circle((node_x, node_y), 1.2 *
                                num_nodes, color=cmap(node.value))
            ax.add_patch(circle)
        # Plot connections
        for node in network.nodes:
            node_angle = node.index * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            for neighbour in node.connections:
                neighbour_index = neighbour.index
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)
                ax.plot((node_x, neighbour_x),
                        (node_y, neighbour_y), color='black')
        im = ax.imshow(np.zeros((1, 1)), interpolation='none', cmap=cmap)
        plt.pause(0.3)  # Pause to allow plot update
    return network


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


def main():
    '''
    This is the main function where all the flags are added so that the user
    may interact with different sections of code, creating different plots
    through the command line.
    '''
    nw = Network()
    # Task 4
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
    # Task 1
    parser.add_argument('-ising_model', action='store_true',
                        help='Run Ising Model')
    parser.add_argument('-test_ising', action='store_true',
                        help='Test Ising Model')
    parser.add_argument('-external', type=float, default=0.0,
                        help='Strength of external opinion')
    parser.add_argument('-alpha', type=float, default=1.0,
                        help='Tolerance parameter for opinion differences.')
    parser.add_argument('-use_network', type=int,
                        help='Uses network for the Ising model.')
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
            nw = ising_with_network(args.use_network, nw)
            nw.plot(default=False)
        else:
            population = np.random.choice([1, -1], size=(100, 100))
            ising_main(population, args.alpha, args.external)

    elif args.network:
        nw.make_random_network(args.network, 0.3)
        nw.plot_network(args.network)

    else:
        print('To get started, please type either: \n'
              '-ring_network <number of nodes> \n'
              '-small_world <number of nodes> \n'
              '-ising_model \n'
              '-test_ising \n'
              '-network <number of nodes> \n'
              '-test_network \n')


if __name__ == "__main__":
    main()
