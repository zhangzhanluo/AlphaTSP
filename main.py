import numpy as np
import copy
from typing import List
import time
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import math

USE_MULTIPROCESSING = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# DEVICE = torch.device('cpu')


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
    input_dim = num_cities * 2 + num_cities ** 2
    model = [nn.Linear(input_dim, 128),
             nn.ReLU(),
             nn.Linear(128, 256),
             nn.ReLU(),
             nn.Linear(256, 256),
             nn.ReLU(),
             nn.Linear(256, 256),
             nn.ReLU(),
             nn.Linear(256, 128),
             nn.ReLU()
             ]
    if policy:
        model.append(nn.Linear(128, num_cities))
        model.append(nn.Softmax(dim=1))
    else:
        model.append(nn.Linear(128, 1))
    model = nn.Sequential(*model)
    return model


class MCTSNode:
    def __init__(self, tsp_env: TSPEnv, parent=None, value=float('-inf'), prior_probability=1.0):
        self.env = tsp_env
        self.parent = parent
        self.children = {}
        self.visit_count = 1
        self.value = value
        self.fully_expanded = False
        self.prior_probability = prior_probability

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
        for action, value_prob in action_values.items():
            env = copy.deepcopy(self.env)
            env.step(action)
            self.children[action] = MCTSNode(env, parent=self, value=value_prob[0], prior_probability=value_prob[1])

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
            policy_input = torch.from_numpy(policy_input).float()
            action_probabilities = self.policy_network(policy_input.to(DEVICE)).squeeze()
            action_probabilities = action_probabilities.detach().cpu().numpy()

            # Masking the probabilities of visited cities to zero
            action_probabilities *= simulated_env.get_unvisited_mask()
            if np.sum(action_probabilities) == 0:
                # If all unvisited cities have probability zero, random selection among unvisited cities
                next_city = np.random.choice(simulated_env.get_unvisited())
            else:
                # Selecting the next city based on the highest probability
                # next_city = int(np.argmax(action_probabilities))
                # Select the next city with probability proportional to the probability
                action_probabilities /= np.sum(action_probabilities)
                next_city = np.random.choice(list(range(simulated_env.num_cities)), p=action_probabilities)
            simulated_env.step(next_city)

        return simulated_env.calculate_total_distance()

    @staticmethod
    def select_child(node):
        non_fully_expanded_children = [child for child in node.children.values() if not child.fully_expanded]
        if len(non_fully_expanded_children) == 0:
            node.fully_expanded = True
            return None
        # Select a child node based on the UCB1 formula
        children_uct = [child.value + child.prior_probability * np.sqrt(np.log(node.visit_count) / child.visit_count)
                        for child in non_fully_expanded_children]
        return non_fully_expanded_children[np.argmax(children_uct)]

    def playout(self):
        node = self.root
        # Selection
        while not node.is_leaf():
            node = self.select_child(node)
            if node is None:  # the node is fully expanded and terminal
                return

        # Expansion
        action_properties = {}  # {action: (value, prior_probability)}
        action_probs = self.policy_network(torch.from_numpy(node.env.get_state()).float().unsqueeze(0).to(DEVICE))
        for action in node.env.get_unvisited():
            action_env = copy.deepcopy(node.env)
            action_env.step(action)
            state_tensor = torch.from_numpy(action_env.get_state()).float().unsqueeze(0)
            action_value = self.value_network(state_tensor.to(DEVICE)).item()
            action_prob = action_probs[0, action].item()
            action_properties[action] = (action_value, action_prob)
        node.expand(action_properties)

        # Simulation
        distance = self.simulate(node.env)

        # Backpropagation
        node.update(-distance)

    def get_best_action(self):
        # return the child node of the root with the highest visit count
        return max(self.root.children, key=lambda action: self.root.children[action].value)

    def get_action(self, n_playouts):
        num_city_left = self.root.env.num_cities - len(self.root.env.current_route)
        num_solution_left = math.factorial(num_city_left)
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
            policy_network = torch.load(policy_network)
        self.policy_network = policy_network.to(DEVICE)
        if isinstance(value_network, str):
            value_network = torch.load(value_network)
        self.value_network = value_network.to(DEVICE)
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
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # float64 -> float32; (120,) -> (1, 120)
            action_probabilities = self.policy_network(state_tensor.to(DEVICE)).squeeze()  # (1,10) -> (10,)
            action_probabilities = action_probabilities.detach().cpu().numpy()  # (10,) -> (10,)
            # Apply mask to zero out probabilities of visited cities
            mask = env.get_unvisited_mask()
            masked_probabilities = action_probabilities * mask

            # Check if all unvisited cities have probability zero
            if np.sum(masked_probabilities) == 0:
                action = np.random.choice(env.get_unvisited())
            else:
                action = np.argmax(masked_probabilities).item()

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
        self.policy_network.to(DEVICE)
        self.value_network.to(DEVICE)
        self.num_playouts_to_get_action_train = num_playouts_to_get_action_train
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
            with mp.Pool() as pool:
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
        states = np.array(training_data['states']).astype(np.float32)  # Convert to float32
        actions = np.array(training_data['actions']).astype(np.float32)  # Convert to float32
        values = np.array(training_data['values']).astype(np.float32)  # Convert to float32

        # Convert to PyTorch tensors
        states_tensor = torch.from_numpy(states)
        actions_tensor = torch.from_numpy(actions).long()
        values_tensor = torch.from_numpy(values)

        action_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_tensor, actions_tensor),
                                                    batch_size=32)
        value_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_tensor, values_tensor),
                                                   batch_size=32)
        self.policy_network.train()
        self.value_network.train()
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(10):
            for states, actions in action_loader:
                states = states.to(DEVICE)
                actions = actions.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.policy_network(states)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
        optimizer = torch.optim.Adam(self.value_network.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(10):
            for states, values in value_loader:
                states = states.to(DEVICE)
                values = values.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.value_network(states).squeeze()
                loss = criterion(outputs, values)
                loss.backward()
                optimizer.step()

    def quick_evaluate_single_env(self, env: TSPEnv):
        agent = AlphaTSP(policy_network=self.policy_network,
                         value_network=self.value_network)

        agent.quick_solve(env)
        distance = env.calculate_total_distance()
        route = env.current_route
        # print('-' * 5 + 'Environment: {}'.format(env.name) + '-' * 5)
        # print('The distance found by the agent: {}'.format(distance))
        # print('The routes found by the agent: {}'.format(route))
        # print('-' * 30)
        return distance, route

    def quick_evaluate(self, envs: List[TSPEnv]):
        # print('<' * 30)
        # print('Evaluating the networks on {} environments'.format(len(envs)))
        start_time = time.time()
        if USE_MULTIPROCESSING:
            with mp.Pool() as pool:
                results = pool.map(self.quick_evaluate_single_env, envs)
        else:
            results = [self.quick_evaluate_single_env(env) for env in envs]
        # Combine results from all environments
        distances, routes = zip(*results)
        print('Time {:.2f}mins; Average distance {:.2f}'.format((time.time() - start_time) / 60,
                                                                  np.mean(distances)))
        # print('>' * 30)
        return distances, routes

    def train(self):
        time_start = time.localtime()
        train_history = []

        # evaluate the initial networks
        print('Evaluate training environments: ', end='')
        train_distances, _ = self.quick_evaluate(self.envs_train)
        print('Evaluate evaluation environments: ', end='')
        eval_distances, _ = self.quick_evaluate(self.envs_eval)
        train_history.append((np.mean(train_distances), np.mean(eval_distances)))

        for iteration in range(self.num_iterations_train):
            start_time = time.time()
            print('Training iteration {}/{}:'.format(iteration + 1, self.num_iterations_train), end=' ')
            training_data = self.collect_mcts_data()
            self.train_networks(training_data)
            print('Time used for training {:.2f}mins'.format((time.time() - start_time) / 60))

            print('Evaluate training environments: ', end='')
            train_distances, _ = self.quick_evaluate(self.envs_train)
            print('Evaluate evaluation environments: ', end='')
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
    plt.close()


if __name__ == '__main__':
    if DEVICE.type == 'cuda':
        print('Using GPU')
        mp.set_start_method('spawn')
    # Generate training and evaluation environments
    N = 20
    num_train_instances = 10
    num_test_instances = 3
    train_instances = [TSPInstance(N, seed=i) for i in range(num_train_instances)]
    test_instances = [TSPInstance(N, seed=i) for i in
                      range(num_train_instances, num_train_instances + num_test_instances)]
    train_envs = [TSPEnv(instance.distance_matrix, instance.current_route, get_mlp_network_input, instance.name) for
                  instance in train_instances]
    test_envs = [TSPEnv(instance.distance_matrix, instance.current_route, get_mlp_network_input, instance.name) for
                 instance in test_instances]

    # training
    trainer = AlphaTSPTrainer(envs_train=train_envs, envs_eval=test_envs, num_iterations_train=5000,
                              num_playouts_to_get_action_train=1000)
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

    # window rolling average
    window_size = 100
    rolling_average_train = []
    rolling_average_test = []
    for i in range(len(distances_train) - window_size):
        rolling_average_train.append(np.mean(distances_train[i:i + window_size]))
        rolling_average_test.append(np.mean(distances_test[i:i + window_size]))
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(rolling_average_train, label='train')
    plt.plot(rolling_average_test, label='test')
    plt.title('Rolling average (window size = {})'.format(window_size))
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.tight_layout()
    fig_name = '{} rolling.png'.format(time.strftime("%m-%d %H-%M", time.localtime(train_start)))
    plt.savefig('Figs/' + fig_name)

    # save model
    torch.save(
        trainer.policy_network,
        'Models/' + 'policy_network_{}.pkl'.format(time.strftime("%m-%d %H-%M", time.localtime(train_start)))
    )
    torch.save(
        trainer.value_network,
        'Models/' + 'value_network_{}.pkl'.format(time.strftime("%m-%d %H-%M", time.localtime(train_start)))
    )
