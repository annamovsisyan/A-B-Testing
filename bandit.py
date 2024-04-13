from abc import ABC, abstractmethod
from logs import *
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    
    """

    This class is for initializing bandit arms.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(): Update the estimated win rate with a new reward value.
    experiment(): Run the experiment..
    report(): Generate a report with statistics about the experiment.
    """

    #@abstractmethod
    def __init__(self, p):
        """
        Initialize the EpsilonGreedy arm.

        Parameters:
        p (float): The win rate of the arm.
        """
        self.p = p
        self.p_estimate = 0 #estimate of average reward
        self.N = 0
        self.r_estimate = 0 #estimate of average regret

    #@abstractmethod
    def __repr__(self):
        
        """
        Return a string representation of the arm.

        Returns:
        str: A string describing the arm.
        """
        return f'An Arm with {self.p} Win Rate'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    #@abstractmethod
    def report(self, N, bandits, chosen_bandit, reward, cumulative_regret, count_suboptimal=None, algorithm = "Epsilon Greedy"):
#TODO ADD DOCSTRING
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        """
        Generate a report with statistics about the experiment.

        Parameters:
        N (int): The number of trials in the experiment.
        bandits (List[str]): List of bandits
	chosen_bandit (np.array): List of chosen bandits
	reward (np.array): Reward of the algorithm
	cumulative_regret (np.array): Cumulative regret of the algorithm
	count_suboptimal Optional[int] : Count of suboptimal pulls, NOTE only for `EpsilonGreedy`
        algorithm (string): Name of the algorithm used.

        Prints:
        Statistics and saves data to CSV files.
        """
        # Save experiment data to a CSV file
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results to a CSV file
        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")
        
        
        print(f"Cumulative Reward : {sum(reward)}")
        
        print(" ")
        
        print(f"Cumulative Regret : {cumulative_regret[-1]}")
              
        print(" ")

        if not count_suboptimal is None:                            
            print(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")


#--------------------------------------#

class Visualization:
    def plot1(self, N, cum_reward_avg, bandits, algorithm='EpsilonGreedy'):
        """
        Visualize the performance of the algorithm in terms of cumulative average reward.
        
        Parameters:
        N (int): The number of trials in the experiment.
        cum_reward_avg (np.array): Cumulative average reward
	bandits (list[Bandits]) : List of Bandit class objects
        algorithm (str): Name of the algorithm used, defaults to 'EpsilonGreedy'.

        Prints:
        Linear and log scale plots of cumulative average reward and optimal reward.
        """
        
        ## LINEAR SCALE
        plt.plot(cum_reward_avg, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("# of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        ## LOG SCALE
        plt.plot(cum_reward_avg, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("# of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, cumulative_rewards_e, cumulative_rewards_t, cumulative_regret_e, cumulative_regret_t):
        """
        Compare Epsilon-Greedy and Thompson Sampling in terms of cumulative rewards and regrets.

        Parameters:
        cumulative_reward_e (np.array): Cumulative rewards results for Epsilon-Greedy
        cumulative_reward_t (np.array): Cumulative rewards results for Thompson-Sampling
        cumulative_regret_e (np.array): Cumulative regret results for Epsilon-Greedy
        cumulative_regret_t (np.array): Cumulative regret results for Thompson-Sampling

        Prints:
        Plots comparing cumulative rewards and cumulative regrets for Epsilon-Greedy and Thompson Sampling.
        """
        ## Cumulative Reward
        plt.plot(cumulative_rewards_e, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_t, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Cumulative Regret
        plt.plot(cumulative_regret_e, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_t, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    
    """
    Epsilon-Greedy multi-armed bandit algorithm.

    This class represents a multi-armed bandit problem solver using the Epsilon-Greedy algorithm.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(x): Update the estimated win rate with a new reward value.
    experiment(BANDIT_REWARDS, N, t=1): Run the experiment..
    report(N, results): Generate a report with statistics about the experiment.
    """

    def __init__(self, p):
        
        """
        Initialize the EpsilonGreedy arm.

        Parameters:
        p (float): The win rate of the arm.
        """
        super().__init__(p)

    def pull(self):
        
        """
        Pull the arm and return the sampled reward.

        Returns:
        float: The sampled reward from the arm.
        """
        return np.random.randn() + self.p

    def update(self, x):
        
        """
        Update the estimated win rate with a new reward value.

        Parameters:
        x (float): The observed reward.
        """
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate

    def experiment(self, true_win_rates, trial_count, time_step=1):
        """
        Run the experiment using Epsilon Greedy Algorithm.
    
        Parameters:
        true_win_rates (list): List of true win rates for each arm.
        trial_count (int): The number of trials.
        time_step (int): Initial time step, defaults to 1.
    
        Returns:
        tuple: Average cumulative reward, cumulative reward, cumulative regret, updated bandits,
               chosen bandit at each trial, reward at each trial, count of suboptimal pulls
        """
        
        # Initialize bandits
        greedy_bandits = [EpsilonGreedy(rate) for rate in true_win_rates]
        rate_means = np.array(true_win_rates)
        optimal_bandit = np.argmax(rate_means)
        suboptimal_pull_count = 0
        epsilon = 1 / time_step
    
        # Arrays to track rewards and bandit choices
        rewards = np.empty(trial_count)
        bandit_choices = np.empty(trial_count)
    
        for i in range(trial_count):
            explore_probability = np.random.random()
            
            if explore_probability < epsilon:
                chosen_index = np.random.choice(len(greedy_bandits))
            else:
                chosen_index = np.argmax([bandit.p_estimate for bandit in greedy_bandits])
    
            reward_obtained = greedy_bandits[chosen_index].pull()
            greedy_bandits[chosen_index].update(reward_obtained)
    
            if chosen_index != optimal_bandit:
                suboptimal_pull_count += 1
            
            rewards[i] = reward_obtained
            bandit_choices[i] = chosen_index
            
            time_step += 1
            epsilon = 1 / time_step
    
        cumulative_avg_reward = np.cumsum(rewards) / (np.arange(trial_count) + 1)
        cumulative_total_reward = np.cumsum(rewards)
        
        cumulative_regrets = np.empty(trial_count)
        for i in range(trial_count):
            cumulative_regrets[i] = trial_count * max(rate_means) - cumulative_total_reward[i]
    
        return (cumulative_avg_reward, cumulative_total_reward, cumulative_regrets, greedy_bandits,
                bandit_choices, rewards, suboptimal_pull_count)



class ThompsonSampling(Bandit):
    """
    ThompsonSampling is a class for implementing the Thompson Sampling algorithm for multi-armed bandit problems.

    Attributes:
    - p (float): The win rate of the bandit arm.
    - lambda_ (float): A parameter for the Bayesian prior.
    - tau (float): A parameter for the Bayesian prior.
    - N (int): The number of times the bandit arm has been pulled.
    - p_estimate (float): The estimated win rate of the bandit arm.

    Methods:
    - pull(): Pull the bandit arm and return the observed reward.
    - sample(): Sample from the posterior distribution of the bandit arm's win rate.
    - update(x): Update the bandit arm's parameters and estimated win rate based on the observed reward.
    - plot(bandits, trial): Plot the probability distribution of the bandit arm's win rate after a given number of trials.
    - experiment(BANDIT_REWARDS, N): Run an experiment to estimate cumulative reward and regret for Thompson Sampling.

    """
    
    def __init__(self, p):
        """
        Initialize a ThompsonSampling bandit arm with the given win rate.

        Parameters:
        p (float): The win rate of the bandit arm.
        """
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1


    def pull(self):
        """
        Pull the bandit arm and return the observed reward.

        Returns:
        float: The observed reward from the bandit arm.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p
    
    def sample(self):
        """
        Sample from the posterior distribution of the bandit arm's win rate.

        Returns:
        float: The sampled win rate from the posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        """
        Update the bandit arm's parameters and estimated win rate based on the observed reward.

        Parameters:
        x (float): The observed reward.
        """
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        
        """
        Plot the probability distribution of the bandit arm's win rate after a given number of trials.

        Parameters:
        bandits (list): List of ThompsonSampling bandit arms.
        trial (int): The number of trials or rounds.

        Displays a plot of the probability distribution of the bandit arm's win rate.

        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, true_win_rates, total_rounds):
        """
        Run an experiment to estimate cumulative reward and regret for Thompson Sampling.
    
        Parameters:
        true_win_rates (list): List of true win rates for each bandit arm.
        total_rounds (int): The number of rounds in the experiment.
    
        Returns:
        tuple: Cumulative reward statistics, updated bandits, and additional experiment data.
        """
        
        thompson_bandits = [ThompsonSampling(rate) for rate in true_win_rates]
    
        checkpoint_rounds = [5, 20, 50, 100, 200, 500, 1000, 1999, 5000, 10000, 19999]
        rewards = np.empty(total_rounds)
        bandit_selections = np.empty(total_rounds)
        
        for round_index in range(total_rounds):
            selected_bandit = np.argmax([bandit.sample() for bandit in thompson_bandits])
    
            if round_index in checkpoint_rounds:
                self.plot(thompson_bandits, round_index)
    
            reward_received = thompson_bandits[selected_bandit].pull()
            thompson_bandits[selected_bandit].update(reward_received)
    
            rewards[round_index] = reward_received
            bandit_selections[round_index] = selected_bandit
    
        cumulative_reward_average = np.cumsum(rewards) / (np.arange(total_rounds) + 1)
        total_cumulative_rewards = np.cumsum(rewards)
        
        total_regrets = np.empty(total_rounds)
        for index in range(total_rounds):
            total_regrets[index] = total_rounds * max(rate.p for rate in thompson_bandits) - total_cumulative_rewards[index]
    
        return cumulative_reward_average, total_cumulative_rewards, total_regrets, thompson_bandits, bandit_selections, rewards


def comparison(N, cumulative_reward_avg_e, cumulative_reward_avg_t, reward_e, reward_t, regret_e, regret_t, bandits):
    # think of a way to compare the performances of the two algorithms VISUALLY 
    
    """
    Compare performance of Epsilon Greedy and Thompson Sampling algorithms in terms of cumulative average reward.

    Parameters:
    N (int): The number of trials in the experiment.
    results_eg (tuple): A tuple of Epsilon Greedy experiment results.
    results_ts (tuple): A tuple of Thompson Sampling experiment results.
    cumulative_reward_avg_e (np.array): Cumulative average reward for EpsilonGreedy
    cumulative_reward_avg_t (np.array): Cumulative average reward for ThompsonSampling
    reward_e (np.array): Reward for EpsilonGreedy
    reward_t (np.array): Reward for ThompsonSampling
    regret_e (np.array): Regret for EpsilonGreedy
    regret_t (np.array): Regret for ThompsonSampling
    
    Prints:
    Linear and log scale plots of cumulative average reward and optimal reward of both algorithms.
    """
    print(f"Total Reward Epsilon Greedy : {sum(reward_e)}")
    print(f"Total Reward Thompson Sampling : {sum(reward_t)}")
        
    print(" ")
        
    print(f"Total Regret Epsilon Greedy : {regret_e}")
    print(f"Total Regret Thompson Sampling : {regret_t}")
        

    plt.figure(figsize=(12, 5))

    ## LINEAR SCALE
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_avg_e, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_t, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")


    ## LOG SCALE
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_avg_e, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_t, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    
    plt.tight_layout()
    plt.show()
    

    
