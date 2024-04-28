import random
import numpy as np
import matplotlib.pyplot as plt


class Node:
    '''Class for nodes.'''

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value


class Network:
    '''Class for network'''

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
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
        self.make_ring_network(N)  # Start with a ring network

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
        plt.set_cmap(cmap)

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes,
                                color=cmap(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x),
                            (node_y, neighbour_y), color='black')

        plt.show()


network = Network()

network.make_random_network(20, connection_probability=0.3)
# network.make_ring_network(20, neighbour_range=2)
# network.make_small_world_network(10)
network.plot()


# def test_networks():
#     '''Tests for networks.'''
#     # Ring network
#     nodes = []
#     num_nodes = 10
#     for node_number in range(num_nodes):
#         connections = [0 for val in range(num_nodes)]
#         connections[(node_number-1) % num_nodes] = 1
#         connections[(node_number+1) % num_nodes] = 1
#         new_node = Node(0, node_number, connections=connections)
#         nodes.append(new_node)
#     network = Network(nodes)

#     print("Testing ring network")
#     assert (network.get_mean_degree() == 2), network.get_mean_degree()
#     assert (network.get_clustering() == 0), network.get_clustering()
#     assert (network.get_path_length() ==
#             2.777777777777778), network.get_path_length()

#     nodes = []
#     num_nodes = 10
#     for node_number in range(num_nodes):
#         connections = [0 for val in range(num_nodes)]
#         connections[(node_number+1) % num_nodes] = 1
#         new_node = Node(0, node_number, connections=connections)
#         nodes.append(new_node)
#     network = Network(nodes)

#     print("Testing one-sided network")
#     assert (network.get_mean_degree() == 1), network.get_mean_degree()
#     assert (network.get_clustering() == 0),  network.get_clustering()
#     assert (network.get_path_length() == 5), network.get_path_length()

#     nodes = []
#     num_nodes = 10
#     for node_number in range(num_nodes):
#         connections = [1 for val in range(num_nodes)]
#         connections[node_number] = 0
#         new_node = Node(0, node_number, connections=connections)
#         nodes.append(new_node)
#     network = Network(nodes)

#     print("Testing fully connected network")
#     assert (network.get_mean_degree() ==
#             num_nodes-1), network.get_mean_degree()
#     assert (network.get_clustering() == 1),  network.get_clustering()
#     assert (network.get_path_length() == 1), network.get_path_length()

#     print("All tests passed")
