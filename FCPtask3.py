import argparse
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm


class Node:
    '''Add docstring.'''

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value


class Network:
    '''Add docstring.'''

    def __init__(self, nodes=None):
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
        # Find neighbors of the current node
        # Calculate the number of neighbors
        # Calculate the total possible connections among neighbors
        # Calculate the actual number of connections among neighbors
        # Calculate clustering coefficient for the current node
        # Update total clustering coefficient
        # Calculate mean clustering coefficient for the entire network

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


# mean_degree = self.get_mean_degree()
# mean_clustering_coefficent = self.get_mean_clustering()
# mean_path_length = self.get_mean_path_length()

    # def make_ring_network(self, N, neighbour_range=1):
    #     # Your code  for task 4 goes here
    #     '''Add docstring.'''

    # def make_small_world_network(self, N, re_wire_prob=0.2):
    #     # Your code for task 4 goes here
    #     '''Add docstring.'''

    def plot(self):
        '''Add docstring.'''

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            # circle = plt.Circle((node_x, node_y), 0.3 *
            #                     num_nodes, color=cm.hot(node.value))
            # ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x),
                            (node_y, neighbour_y), color='black')


def main():  # main fucntion
    '''Add docstring.'''
    parser = argparse.ArgumentParser()  # use argpase
    parser.add_argument("-network", type=int, default=10,
                        help="size of network")
    parser.add_argument("-test_newtworks", action='store_true', default=False)
    args = parser.pase_args()
    network = network()
    network = network.make_random_network(args.network)
    print("Mean degree:", mean_degree)
    print("Mean clustering coefficient:", mean_clustering_coefficent)
    print("Mean path length:", mean_path_length)
    if args.test_networks == True:
        test_networks()


def test_networks():
    '''Add docstring.'''

    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1) % num_nodes] = 1
        connections[(node_number+1) % num_nodes] = 1
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
