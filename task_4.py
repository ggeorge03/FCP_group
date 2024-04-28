import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Node:
    '''Class for nodes.'''

    def __init__(self, value, number, connections=None):
        '''Initialises the nodes'''
        self.index = number
        self.connections = connections
        self.value = value


class Network:
    '''Class for network.'''

    def __init__(self, nodes=None):
        '''Initialises the network.'''
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

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
                                for j in range(1, neighbour_range + 1)]

    def make_small_world_network(self, N, re_wire_prob=0.2):
        '''
        This function creates a small world network that starts with a ring
        network and then rewires edges.

        N = number of nodes in the network.

        re_wire_prob = the probability of rewiring an edge. Deafult is 0.2.
        '''
        self.make_ring_network(N)  # Start with a ring network.

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

    def node_degree(self):
        '''Obtains the degree of each node.'''
        node_degrees = []
        for node in self.nodes:
            degree = sum(node.connections)
            node_degrees.append(degree)
        return node_degrees

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


def test_network():
    network = Network()

    # Test make_random_network
    network.make_random_network(10)
    assert len(
        network.nodes) == 10, "make_random_network did not create the correct number of nodes"

    # Test make_ring_network
    network.make_ring_network(10)
    assert len(
        network.nodes) == 10, "make_ring_network did not create the correct number of nodes"

    # Test make_small_world_network
    network.make_small_world_network(10)
    assert len(
        network.nodes) == 10, "make_small_world_network did not create the correct number of nodes"

    # Test node_degree
    # degrees = network.node_degree()
    # assert len(degrees) == len(
    #     network.nodes), "node_degree did not return the correct number of degrees"

    print('All tests passed successfully')


def main():
    '''g'''
    network = Network()
    test_network()

    parser = argparse.ArgumentParser(
        description='Create plots for a ring network or a small world network.')
    parser.add_argument('-ring_network', type=int,
                        help='Creates ring network with inputted number of nodes')
    parser.add_argument('-small_world', type=int,
                        help='Creates small world network with inputted number of nodes')
    parser.add_argument('-re_wire', type=float, default=0.2,
                        help='Probability of rewiring for small world network')

    args = parser.parse_args()

    if args.ring_network:
        network.make_ring_network(args.ring_network)
        network.plot()

    elif args.small_world:
        network.make_small_world_network(
            args.small_world, re_wire_prob=args.re_wire)
        network.plot()

    else:
        print('Type either -ring_network <number of nodes> or -small_world <number of nodes> to create a network.')


if __name__ == "__main__":
    main()
