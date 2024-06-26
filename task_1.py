import argparse
import numpy as np
import matplotlib.pyplot as plt


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
<<<<<<< HEAD
    ] #define neighbours(above, below, right, left) 
=======
    ]  # define neighbours(above, below, right, left)
>>>>>>> c081995ecdc7e4cf451df2b882da180c2b3c0fb5
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


def main():  # Function to allow user to test if code is correct.
    '''
    Main function allows the user to run and test the Ising model.
    Command-line args:
        -ising_model runs the Ising model simulation
        -test_ising tests the Ising model
        -external strength of external opinion, default = 0.0
        -alpha tolerance parameter for opinion differences, default = 1.0.
    '''
    parser = argparse.ArgumentParser(description="Run Ising model")
    parser.add_argument('-ising_model', action='store_true',
                        help='Run Ising Model')
    parser.add_argument('-test_ising', action='store_true',
                        help='Test Ising Model')
    parser.add_argument('-external', type=float, default=0.0,
                        help='Strength of external opinion')
    parser.add_argument('-alpha', type=float, default=1.0,
                        help='Tolerance parameter for opinion differences.')
    args = parser.parse_args()
    population = np.random.choice([1, -1], size=(100, 100))
    if args.test_ising:
        test_ising()
    if args.ising_model:
        ising_main(population, args.alpha, args.external)


if __name__ == "__main__":
    main()
