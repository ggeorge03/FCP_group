import argparse
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
import random


class Node:
    '''Class to represent a node in the network'''

    def __init__(self, value, number, connections=None):
        '''Function that initialises a node
        value = value associated with the node,
        number = the index of the node,
        connections = a list to represent the connections between one node to another,
        with default set to None'''
=======
# import matplotlib.cm as cm


class Node:
    '''Add docstring.'''

    def __init__(self, value, number, connections=None):

>>>>>>> 696c40a4c210afdb2f6fcf9d4b292df87250620e
        self.index = number
        self.connections = connections
        self.value = value


class Network:
<<<<<<< HEAD
    '''Class that represents a network of nodes.'''

    def __init__(self, nodes=None):
        '''Function that initialises a network
        nodes = a list of notes in the network, with default set to None'''
=======
    '''Add docstring.'''

    def __init__(self, nodes=None):
>>>>>>> 696c40a4c210afdb2f6fcf9d4b292df87250620e
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
<<<<<<< HEAD
        '''Function that is calculating the mean degree of a network, summing up
        all of the connections of all the nodes and dividing by the number of nodes'''

        total_degree = sum(sum(node.connections) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        '''Function that is calculating the mean clustering coefficient of the network'''

        total_clustering_coefficient = 0
        for node in self.nodes:
            neighbours = [self.nodes[i] for i, conn in enumerate(node.connections) if conn]
            num_of_neighbours = len(neighbours)
            if num_of_neighbours < 2:
                continue
            possible_connections = num_of_neighbours * (num_of_neighbours - 1) / 2
            actual_connections = 0
            for i in range(num_of_neighbours):
                for c in range(i + 1, num_of_neighbours):
                    if node.connections[neighbours[i].index] and node.connections[neighbours[c].index]:
                        actual_connections += 1
            clustering_coefficient = actual_connections / possible_connections if possible_connections != 0 else 0
            total_clustering_coefficient += clustering_coefficient

        mean_clustering_coefficient = total_clustering_coefficient / len(self.nodes)

        return mean_clustering_coefficient

    def get_mean_path_length(self):
        '''Function that is calculating the mean path length of the network'''

        total_path_length = 0

        for node in self.nodes:
            distances = self.bfs(node)
            total_path_length += sum(distances.values())

        mean_path_length = total_path_length / (len(self.nodes) * (len(self.nodes) - 1))

        return mean_path_length

    def bfs(self, start_node):
        '''Function that undergoes a breadth first search from a start node.'''

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
        Each node is connected to each other node with probability p
        N = the number of nodes in a network
        connection_probability = the probability of connections between nodes
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()  #this is to generate a random float between 0 and 1 to represent the value associated with a node
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        '''add in to make sure every node is connected to at least one other node'''
        for index, node in enumerate(self.nodes):
            other_node_index = random.choice([i for i in range(N) if i != index])
            node.connections[other_node_index] = 1
            self.nodes[other_node_index].connections[index] = 1

        for index, node in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

        mean_degree = self.get_mean_degree()
        mean_clustering_coefficent = self.get_mean_clustering()
        mean_path_length = self.get_mean_path_length()

        print("Mean degree:", mean_degree)
        print("Mean clustering coefficient:", mean_clustering_coefficent)
        print("Mean path length:", mean_path_length)

        coordinates_of_nodes = {node: (np.random.uniform(0, N), np.random.uniform(0, N)) for node in range(N)}

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

    def test_network(self):
        '''Test Function - for networks'''

        nodes = []
        num_nodes = 10
        for node_number in range(num_nodes):
            connections = [1 for _ in range(num_nodes)]
            connections[node_number] = 0
            new_node = Node(0, node_number, connections=connections)
            nodes.append(new_node)
        network = Network(nodes)

        print("Testing fully connected network")

        assert network.get_mean_degree() == num_nodes - 1, network.get_mean_degree()
        assert network.get_mean_clustering() == 1, network.get_mean_clustering()
        assert network.get_mean_path_length() == 1, network.get_mean_path_length()

        print("All tests passed")


def main():
    '''This is a main function that uses argparse'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", type=int, default=10, help="size of network")
    parser.add_argument("-test_network", action='store_true', default=False)
    args = parser.parse_args()
    if args.test_network:
        Network.test_network()
    elif args.network:
        network = Network()
        network.make_random_network(args.network, 0.3)


if __name__ == "__main__":
    main()
=======
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
>>>>>>> 696c40a4c210afdb2f6fcf9d4b292df87250620e
