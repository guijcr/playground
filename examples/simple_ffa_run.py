"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents

def main():
    """Simple function to bootstrap a game.
       Use this as an example to set up your training env.
    """
    for x in range(220):
        # Print all possible environments in the Pommerman registry
        print(pommerman.REGISTRY)

        # Create a set of agents (exactly four)
        agent_list = [
            agents.AgressiveAgent(),
            agents.SupSafeAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
        print(env.action_space.n)  # number of actions the environment has
        #print("Original list is : " + str(agent_list))
        #print("List index-value are : ")
        #for i in range(len(agent_list)):
        #    print(i, end=" ")
        #    print(agent_list[i])
            # Run the episodes just like OpenAI Gym
        for i_episode in range(1):
            state = env.reset()
            done = False
            while not done:
                env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
            print('Episode {} finished'.format(i_episode))
            with open('Rewards.txt', 'a') as file:              # Code written by Guilherme Rodrigues - 1644650
                file.write(f'{reward}\n')                       # This code written from line 39 to line 45 it is
            # with open('outfile2.txt', 'a') as file:           # to print the results of each run in a text file
            #     file.write(f'{state}\n')                      # called "Results" in the Examples folder
            # with open('outfile3.txt', 'a') as file:
            #     file.write(f'{done}\n')
            with open('Results.txt', 'a') as file:
                file.write(f'{info}\n')                         # Code written by G. Rodrigues ends here
        env.close()


if __name__ == '__main__':
    main()
