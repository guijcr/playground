"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents
import numpy as np  # QLearningExperiment
import matplotlib.pyplot as plt


def main():
    """Simple function to bootstrap a game.
       Use this as an example to set up your training env.
    """
    for x in range(1):
        # Print all possible environments in the Pommerman registry
        print(pommerman.REGISTRY)

        # Create a set of agents (exactly four)
        agent_list = [
            agents.SimpleAgent(),
            agents.MyAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

        learning_rate = 0.4  # QLearningExperiment
        discount = 0.95  # measure of how important we find/value future actions/reward over current actions/reward
        episodes = 50  # QLearningExperiment
        #print(env.observation_space.high)  # values - [0.6  0.07]
        #print(env.observation_space.low)  #values - [-1.2  -0.07]
        show_every = 500  # QLearningExperiment

        discrete_os_size = [20] * len(env.observation_space.high)
        discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size


        epsilon = 0.5  #this will be used as a random variable so the higher the value the more random
        start_epsilon_decaying = 1
        end_epsilon_decaying = episodes // 2  # the double // is to make sure it is an intenger and not a float

        epsilon_decaying_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

        #print(env.action_space.n)  # number of actions the environment has #QLearningExperiment
        #print(env.observation_space.high)  # QLearningExperiment
        #print(env.observation_space.low)  # QLearningExperiment
        #print(discrete_os_size)
        q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))
        #q_table = np.arange(370.0).reshape(10, 10, 10)

        #aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

        def get_discrete_state(state):
            discrete_state = (state - env.observation_space.low) / discrete_os_win_size
            return tuple(discrete_state.astype(np.int))

        # Run the episodes just like OpenAI Gym
        for i_episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            if i_episode % show_every == 0:
                print(i_episode)
                render = True
            else:
                render = False
            discrete_state = get_discrete_state(env.reset())
            done = False
            while not done:
                if np.random.randint(0, 5) > epsilon:
                    action = np.argmax(q_table[discrete_state])
                else:
                    action = np.random.randint(0, env.action_space.n)
                new_state, reward, done, _ = env.step(action)
                episode_reward += reward
                new_discrete_state = get_discrete_state(new_state)
                # print(reward, new_state)
                if render:
                    env.render()
                if not done:
                    max_future_q = np.max(q_table[new_discrete_state])
                    current_q = q_table[discrete_state + (action,)]
                    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                    q_table[discrete_state + (action,)] = new_q
                elif new_state[0] >= env.unwrapped.goal_position:
                    print(f"We made it on episode {i_episode}")
                    q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            if end_epsilon_decaying >= i_episode >= start_epsilon_decaying:
                epsilon -= epsilon_decaying_value

            ep_rewards.append(episode_reward)
            if not i_episode % show_every:
                # work on our dictionary
                average_reward = sum(ep_rewards[-show_every:]) / show_every
                aggr_ep_rewards['ep'].append(i_episode)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
                aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))
                print(
                    f"Episode: {i_episode} avg: {average_reward} min: {min(ep_rewards[-show_every:])} max: {max(ep_rewards[-show_every:])}")
                # this print will give us the metrics
                np.save(f"qtables/{i_episode}-qtable.pny", q_table)  # saving the qtable in the qtables directory

        env.close()

        # The x axis will always be the episodes
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], Label="avg")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], Label="min")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], Label="max")
        plt.legend(loc=4)  # location of the legend for the graph
        plt.show()


if __name__ == '__main__':
    main()
