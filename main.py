import numpy as np
import tensorflow as tf
import copy
from typing import List
import time
import multiprocessing
from matplotlib import pyplot as plt

USE_MULTIPROCESSING = False


class TSPInstance:
    def __init__(self, num_cities: int, seed: int = None):
        self.num_cities = num_cities
        self.rng = np.random.RandomState(seed)
        self.distance_matrix = self._generate_distance_matrix()
        self.current_route = (self.rng.randint(0, self.num_cities),)
        self.name = 'TSP-{}-{}'.format(self.num_cities, seed)

    def _generate_distance_matrix(self):
        # Generate a random distance matrix
        # The distance matrix is symmetric with zeros on the diagonal
        # The distances are integers in the range [1, 100]
        distance_matrix = self.rng.randint(1, 100, size=(self.num_cities, self.num_cities))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix


class TSPEnv:
    def __init__(self, distance_matrix: np.ndarray, current_route: tuple, state_representation, name=None):
        self.distance_matrix = distance_matrix
        self.current_route = current_route
        self.initial_route = current_route
        self.num_cities = distance_matrix.shape[0]
        self.state_representation = state_representation
        self.name = name if name else 'TSP-{}'.format(self.num_cities)

    def __repr__(self):
        return 'Name: {}, City: {}, Current Route: {}'.format(
            self.name,
            self.num_cities,
            self.current_route)

    def reset(self):
        self.current_route = self.initial_route

    def get_state(self):
        return self.state_representation(self)

    def get_unvisited_mask(self):
        visited_mask = np.ones(self.num_cities)
        visited_mask[list(self.current_route)] = 0
        return visited_mask

    def get_unvisited(self) -> list:
        return [city for city in range(self.num_cities) if city not in self.current_route]

    def is_done(self):
        return len(self.current_route) == self.num_cities

    def step(self, next_city: int):
        assert next_city not in self.current_route
        self.current_route += (next_city,)
        return self.get_state()

    def calculate_total_distance(self):
        # Calculate the total distance of the tour represented by 'solution'
        # Assuming 'solution' is a list of city indices in the order they are visited
        total_distance = sum(self.distance_matrix[self.current_route[i], self.current_route[i + 1]] for i in
                             range(len(self.current_route) - 1))
        # Add distance from last city back to the first city
        total_distance += self.distance_matrix[self.current_route[-1], self.current_route[0]]
        return total_distance

    def solve(self, strategy):
        self.reset()
        while not self.is_done():
            next_city = strategy(self)
            self.step(next_city)
        return self.calculate_total_distance()


def get_mlp_network_input(tsp_env: TSPEnv):
    """State representation for the MLP network"""
    # One-hot encoding of the current city
    current_city_one_hot = np.zeros(tsp_env.distance_matrix.shape[0])
    current_city_one_hot[tsp_env.current_route[-1]] = 1

    # One-hot encoding of the visited cities
    visited_mask = np.zeros(tsp_env.distance_matrix.shape[0])
    visited_mask[list(tsp_env.current_route)] = 1

    # Concatenate the one-hot encodings with the distance matrix
    network_input = np.concatenate([
        current_city_one_hot,
        visited_mask,
        tsp_env.distance_matrix.flatten()
    ])

    return network_input


def build_mlp_network(num_cities, policy=True):
    # Input layers
    input_shape = (num_cities * 2 + num_cities ** 2,)
    inputs = tf.keras.Input(shape=input_shape)

    # Hidden layers
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Output layers
    if policy:
        outputs = tf.keras.layers.Dense(num_cities, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    else:
        outputs = tf.keras.layers.Dense(1)(x)
        loss = 'mse'

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss)

    return model


class MCTSNode:
    def __init__(self, tsp_env: TSPEnv, parent=None, value=float('-inf')):
        self.env = tsp_env
        self.parent = parent
        self.children = {}
        self.visit_count = 1
        self.value = value
        self.fully_expanded = False

    def __repr__(self):
        return 'Visit: {}, Children: {}, Value: {:.2f}, Route: {}'.format(self.visit_count,
                                                                          len(self.children.keys()),
                                                                          self.value,
                                                                          self.env.current_route)

    def is_leaf(self):
        return not self.children

    def expand(self, action_values):
        if len(action_values) == 0:
            self.fully_expanded = True
            return
        # Expand by adding one child per action
        for action, value in action_values.items():
            env = copy.deepcopy(self.env)
            env.step(action)
            self.children[action] = MCTSNode(env, parent=self, value=value)

    def update(self, distance_negative):
        self.visit_count += 1
        if self.visit_count == 2:
            self.value = distance_negative
        else:
            self.value = max(self.value, distance_negative)
        if self.parent:
            self.parent.update(distance_negative)


class MonteCarloTreeSearch:
    def __init__(self, policy_network, value_network, tsp_env: TSPEnv):
        self.policy_network = policy_network
        self.value_network = value_network
        self.root = MCTSNode(tsp_env)

    def __repr__(self):
        return 'Route: {}, Distance: {}'.format(self.root.env.current_route, -self.root.value)

    def simulate(self, tsp_env: TSPEnv):
        """Simulate a play-out from the given state using the policy network to choose actions."""
        simulated_env = copy.deepcopy(tsp_env)

        while not simulated_env.is_done():
            # Format the input for the policy network
            policy_input = simulated_env.get_state().reshape(1, -1)
            action_probabilities = self.policy_network.predict(policy_input, verbose=0)[0]

            # Masking the probabilities of visited cities to zero
            action_probabilities *= simulated_env.get_unvisited_mask()
            if np.sum(action_probabilities) == 0:
                # If all unvisited cities have probability zero, random selection among unvisited cities
                next_city = np.random.choice(simulated_env.get_unvisited())
            else:
                # Selecting the next city based on the highest probability
                next_city = int(np.argmax(action_probabilities))
            simulated_env.step(next_city)

        return simulated_env.calculate_total_distance()

    @staticmethod
    def select_child(node):
        non_fully_expanded_children = [child for child in node.children.values() if not child.fully_expanded]
        if len(non_fully_expanded_children) == 0:
            node.fully_expanded = True
            return None
        # Select a child node based on the UCB1 formula
        children_uct = [child.value + 2 * np.sqrt(np.log(node.visit_count) / child.visit_count) for child in
                        non_fully_expanded_children]
        return non_fully_expanded_children[np.argmax(children_uct)]

    def playout(self):
        node = self.root
        # Selection
        while not node.is_leaf():
            node = self.select_child(node)
            if node is None:  # the node is fully expanded and terminal
                return

        # Expansion
        action_values = {}
        for action in node.env.get_unvisited():
            action_env = copy.deepcopy(node.env)
            action_env.step(action)
            action_values[action] = self.value_network.predict(action_env.get_state().reshape(1, -1), verbose=0)[0][0]
        node.expand(action_values)

        # Simulation
        distance = self.simulate(node.env)

        # Backpropagation
        node.update(-distance)

    def get_best_action(self):
        # return the child node of the root with the highest visit count
        return max(self.root.children, key=lambda action: self.root.children[action].value)

    def get_action(self, n_playouts):
        num_city_left = self.root.env.num_cities - len(self.root.env.current_route)
        num_solution_left = np.math.factorial(num_city_left)
        for _ in range(min(n_playouts, num_solution_left)):
            self.playout()
        return self.get_best_action()

    def reset_tree(self, action):
        self.root = self.root.children[action]
        self.root.parent = None


class AlphaTSP:
    def __init__(self,
                 policy_network,
                 value_network,
                 num_playouts_to_get_action=1000,
                 ):
        if isinstance(policy_network, str):
            policy_network = tf.keras.models.load_model(policy_network)
        self.policy_network = policy_network
        if isinstance(value_network, str):
            value_network = tf.keras.models.load_model(value_network)
        self.value_network = value_network
        self.num_playouts_to_get_action = num_playouts_to_get_action

    def solve(self, env: TSPEnv):
        env.reset()
        state = env.get_state()
        mcts = MonteCarloTreeSearch(policy_network=self.policy_network,
                                    value_network=self.value_network,
                                    tsp_env=env)
        states, actions = [], []
        while not env.is_done():
            action = mcts.get_action(self.num_playouts_to_get_action)
            states.append(state)
            actions.append(action)
            state = env.step(action)
            mcts.reset_tree(action)
        return states, actions

    def quick_solve(self, env: TSPEnv):
        env.reset()
        state = env.get_state()
        while not env.is_done():
            action_probabilities = self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]
            action_probabilities *= env.get_unvisited_mask()
            if np.sum(action_probabilities) == 0:
                # If all unvisited cities have probability zero, random selection among unvisited cities
                action = np.random.choice(env.get_unvisited())
            else:
                # Selecting the next city based on the highest probability
                action = int(np.argmax(action_probabilities))
            env.step(action)
        return env.calculate_total_distance()


class AlphaTSPTrainer:
    def __init__(self,
                 envs_train: List[TSPEnv] = None,  # List of training environments
                 envs_eval: List[TSPEnv] = None,  # List of evaluation environments
                 num_playouts_to_get_action_train=100,  # Number of playouts to run MCTS for getting an action
                 num_playouts_to_get_action_eval=1000,  # Number of playouts to run MCTS for getting an action
                 num_iterations_train=50,  # Number of training iterations
                 ):
        self.envs_train = envs_train
        self.envs_eval = envs_eval
        self.policy_network = build_mlp_network(envs_train[0].num_cities, policy=True)
        self.value_network = build_mlp_network(envs_train[0].num_cities, policy=False)
        self.num_playouts_to_get_action_train = num_playouts_to_get_action_train
        self.num_playouts_to_get_action_eval = num_playouts_to_get_action_eval
        self.num_iterations_train = num_iterations_train

    def collect_mcts_data_for_env(self, env: TSPEnv):
        agent = AlphaTSP(policy_network=self.policy_network,
                         value_network=self.value_network,
                         num_playouts_to_get_action=self.num_playouts_to_get_action_train)
        states, actions = agent.solve(env)
        distance = env.calculate_total_distance()
        training_data = {
            'states': states,
            'actions': actions,
            'values': [-distance for _ in range(len(states))]
        }
        return training_data

    def collect_mcts_data(self):
        if USE_MULTIPROCESSING:
            with multiprocessing.Pool() as pool:
                results = pool.map(self.collect_mcts_data_for_env, self.envs_train)
        else:
            results = [self.collect_mcts_data_for_env(env) for env in self.envs_train]
        combined_results = {
            'states': [],
            'actions': [],
            'values': []
        }
        for result in results:
            combined_results['states'].extend(result['states'])
            combined_results['actions'].extend(result['actions'])
            combined_results['values'].extend(result['values'])
        return combined_results

    def train_networks(self, training_data):
        states = np.array(training_data['states'])
        actions = tf.keras.utils.to_categorical(training_data['actions'], num_classes=self.envs_train[0].num_cities)
        values = np.array(training_data['values'])

        self.policy_network.fit(states, actions, epochs=10, batch_size=32, verbose=0)
        self.value_network.fit(states, values, epochs=10, batch_size=32, verbose=0)

    def evaluate_single_env(self, env: TSPEnv):
        agent = AlphaTSP(policy_network=self.policy_network,
                         value_network=self.value_network,
                         num_playouts_to_get_action=self.num_playouts_to_get_action_eval)

        agent.quick_solve(env)
        distance = env.calculate_total_distance()
        route = env.current_route
        print('-' * 5 + 'Environment: {}'.format(env.name) + '-' * 5)
        print('The distance found by the agent: {}'.format(distance))
        print('The routes found by the agent: {}'.format(route))
        print('-' * 30)
        return distance, route

    def quick_evaluate(self, envs: List[TSPEnv]):
        print('<' * 30)
        print('Evaluating the networks on {} environments'.format(len(envs)))
        start_time = time.time()
        if USE_MULTIPROCESSING:
            with multiprocessing.Pool() as pool:
                results = pool.map(self.evaluate_single_env, envs)
        else:
            results = [self.evaluate_single_env(env) for env in envs]
        # Combine results from all environments
        distances, routes = zip(*results)
        print('Average distance on environments: {:.2f}'.format(np.mean(distances)))
        print('Time used for evaluation: {:.2f}mins'.format((time.time() - start_time) / 60))
        print('>' * 30)
        return distances, routes

    def train(self):
        time_start = time.localtime()
        train_history = []

        # evaluate the initial networks
        print('Evaluating the initial networks on training environments')
        train_distances, _ = self.quick_evaluate(self.envs_train)
        print('Evaluating the initial networks on evaluation environments')
        eval_distances, _ = self.quick_evaluate(self.envs_eval)
        train_history.append((np.mean(train_distances), np.mean(eval_distances)))

        for iteration in range(self.num_iterations_train):
            start_time = time.time()
            print('Training iteration {}/{}'.format(iteration + 1, self.num_iterations_train))
            training_data = self.collect_mcts_data()
            self.train_networks(training_data)
            print('Time used for training: {:.2f}mins'.format((time.time() - start_time) / 60))

            print('Evaluating the networks on training environments')
            train_distances, _ = self.quick_evaluate(self.envs_train)
            print('Evaluating the networks on evaluation environments')
            test_distances, _ = self.quick_evaluate(self.envs_eval)
            train_history.append((np.mean(train_distances), np.mean(test_distances)))
            plot_train_records(train_history, time_start)
        return train_history


def plot_train_records(train_history, time_start):
    d_train, d_test = zip(*train_history)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(d_train, label='train')
    plt.plot(d_test, label='test')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.tight_layout()
    figure_name = '{}.png'.format(time.strftime("%m-%d %H-%M", time_start))
    plt.savefig('Figs/' + figure_name)
    plt.show()


if __name__ == '__main__':
    # Generate training and evaluation environments
    N = 10
    num_train_instances = 5
    num_test_instances = 3
    train_instances = [TSPInstance(N, seed=i) for i in range(num_train_instances)]
    test_instances = [TSPInstance(N, seed=i) for i in
                      range(num_train_instances, num_train_instances + num_test_instances)]
    train_envs = [TSPEnv(instance.distance_matrix, instance.current_route, get_mlp_network_input, instance.name) for
                  instance in train_instances]
    test_envs = [TSPEnv(instance.distance_matrix, instance.current_route, get_mlp_network_input, instance.name) for
                 instance in test_instances]

    # training
    trainer = AlphaTSPTrainer(envs_train=train_envs, envs_eval=test_envs, num_iterations_train=50)
    train_start = time.time()
    train_records = trainer.train()
    train_end = time.time()

    # evaluation
    solver = AlphaTSP(policy_network=trainer.policy_network,
                      value_network=trainer.value_network,
                      num_playouts_to_get_action=1000)

    final_distances_train = []
    for train_env in train_envs:
        d, _ = solver.solve(train_env)
        final_distances_train.append(train_env.calculate_total_distance())
    final_distances_test = []
    for test_env in test_envs:
        d, _ = solver.solve(test_env)
        final_distances_test.append(test_env.calculate_total_distance())

    # plot result
    distances_train, distances_test = zip(*train_records)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(distances_train, label='train')
    plt.plot(distances_test, label='test')
    plt.title('Time used: {:.2f}mins; \nTrain performance: {:.2f}; \nTest performance: {:.2f}'.format(
        (train_end - train_start) / 60,
        np.mean(final_distances_train),
        np.mean(final_distances_test)))
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.tight_layout()
    fig_name = '{}.png'.format(time.strftime("%m-%d %H-%M", time.localtime(train_start)))
    plt.savefig('Figs/' + fig_name)
    plt.show()
