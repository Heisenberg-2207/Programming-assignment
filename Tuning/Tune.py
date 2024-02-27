from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import Algorithms


def sarsa_softmax_tune():
 pbounds = {'alpha': (0, 1), 'tau': (0.1, 1)}
 alpha_sfm_sarsa = []
 tau_sarsa = []

 index = 1
 episodes = 5000
 gamma = 0.95
 epsilon = 0.1

 for index in range(0,6):
    env  = Algorithms.world(world_num = index)
    Problem = Algorithms.Solver(env,episodes,index)
    Problem.initiate(Problem.choose_action_softmax,Problem.sarsa)
  

    optimizer = BayesianOptimization(
     f=Problem.solve_softmax,
     pbounds=pbounds,
     random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=4)
    alpha_sfm_sarsa.append(optimizer.max['params']['alpha'])
    tau_sarsa.append(optimizer.max['params']['tau'])
    cost = Problem.solve_softmax(optimizer.max['params']['alpha'],optimizer.max['params']['tau'])
    steps = Problem.getSteps()
 return alpha_sfm_sarsa,tau_sarsa, cost, steps

def sarsa_eps_tune():
 pbounds = {'alpha': (0, 1), 'epsilon': (0.1, 1)}
 alpha_eps_sarsa = []
 esp_sarsa = []

 index = 1
 episodes = 5000
 gamma = 0.95
 epsilon = 0.1

 for index in range(0,6):
    env  = Algorithms.world(world_num = index)
    Problem = Algorithms.Solver(env,episodes,index)
    Problem.initiate(Problem.choose_action_epsilon,Problem.sarsa)
  

    optimizer = BayesianOptimization(
     f=Problem.solve_eps_greedy,
     pbounds=pbounds,
     random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=4)
    alpha_eps_sarsa.append(optimizer.max['params']['alpha'])
    esp_sarsa.append(optimizer.max['params']['epsilon'])
    cost = Problem.solve_softmax(optimizer.max['params']['alpha'],optimizer.max['params']['epsilon'])
    steps = Problem.getSteps()
 return alpha_eps_sarsa,esp_sarsa, cost, steps

def q_sfm_tune():
 pbounds = {'alpha': (0, 1), 'tau': (0.1, 1)}
 alpha_sfm_q = []
 tau_q = []

 index = 1
 episodes = 5000
 gamma = 0.95
 epsilon = 0.1

 for index in range(0,6):
    env  = Algorithms.world(world_num = index)
    Problem = Algorithms.Solver(env,episodes,index)
    Problem.initiate(Problem.choose_action_softmax,Problem.qlearning)
  

    optimizer = BayesianOptimization(
     f=Problem.solve_softmax,
     pbounds=pbounds,
     random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=4)
    alpha_sfm_q.append(optimizer.max['params']['alpha'])
    tau_q.append(optimizer.max['params']['tau'])
    cost = Problem.solve_softmax(optimizer.max['params']['alpha'],optimizer.max['params']['tau'])
    steps = Problem.getSteps()
 return alpha_sfm_q,tau_q,cost, steps

def q_eps_tune():
 pbounds = {'alpha': (0, 1), 'epsilon': (0.1, 1)}
 alpha_eps_q = []
 esp_q = []

 index = 1
 episodes = 5000
 gamma = 0.95
 epsilon = 0.1

 for index in range(0,6):
    env  = Algorithms.world(world_num = index)
    Problem = Algorithms.Solver(env,episodes,index)
    Problem.initiate(Problem.choose_action_epsilon,Problem.qlearning)
  

    optimizer = BayesianOptimization(
     f=Problem.solve_eps_greedy,
     pbounds=pbounds,
     random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=4)
    alpha_eps_q.append(optimizer.max['params']['alpha'])
    esp_q.append(optimizer.max['params']['epsilon'])
    cost = Problem.solve_softmax(optimizer.max['params']['alpha'],optimizer.max['params']['epsilon'])
    steps = Problem.getSteps()
 return alpha_eps_q, esp_q, cost, steps
