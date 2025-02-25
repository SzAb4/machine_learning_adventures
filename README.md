In this project, I am playing around with different machine learning algorithms and toy examples. Here is a list of them:

# Deep Q-learning 

## The game
We have a 10 by 10 grid, in which the agent starts at the left-top position. At the bottom-right of the grid, there is a reward. 
Each step, the agent has to decide between four actions: Right, Down, Left and Up. For each move made, there would be a penalty (a negative reward). 
When the agent reaches the field with the reward, the game ends. The goal is to reach it as quickly as possible. To make it a bit more complex, I added
obstacles, meaning that there were some squares that the agent would not be allowed to go to. Of course this is an extremely simple game,
but I thought it was a good starting point. 

## The training setup
I made the agent play a number of games. After each step made, there would be 
a backpropagation step through the network (called Critic) predicting the Q-function. The moves of the agent where programmed to first be random (in order to explore the environment),
and over time be more and more according to the output of the neural net.
I would fix the number of epochs of training, where one epoch was one game played. 

## Results
The first times I ran it, it would always take an incredibly long time to run. I looked a t the behaviour in detail and made a few observations:
- The first thing that the network learns is in the very last step before winning the game: When it wins, this gives a strong reinforcement
to the last move. However, it seemed that it could unlearn this again: If the agent takes a long time to reach the goal, the neural net would
converge to approximating a constant function, and so undoing the nudge it got for winning the game in previous iterations. I fiddled around a little,
and after making several adjustments to the following parameters I got it to work:
- The learning rate: Of course this is one of the most important hyper-parameters, it looks like to me that there is a certain range of
1-2 orders of magnitude of learning rates that work.
- How high the reward should be: First I started by giving only a very small reward at the end. By increasing the reward, there would
be a stronger reinforcement for making the right move at the end.
- The epsilon: Epsilon is the parameter that determines how much the actor moves according to the Q-function: First, epsilon is 1, meaning
that every move is random. Over time, it would decrease, meaning that there was a higher and higher chance, that the actor would choose 
the best move according to the Q-function. Oftentimes, the epsilon would decrease too fast, meaning that it would stop exploring before
the Q-function was in any way meaningful, and sometimes get stuck in loops (presumably), making training last forever. 

## Takeaways
- The structure of the rewards is quite important. At least for this basic algorithm, it makes a big difference what I reward it at the end,
even though it makes no difference to the game and what the best moves are
- The tradeoff between exploring and following the already known policy is something to pay attention to
- Some things I got to work through trial and error. I wonder how people at big labs do these things, when they train a big LLM it has to 
work on the first try. The number of possible combinations for hyperparameters becomes huge very quickly, somehow there must be a process
of building intuition of which hyperparameters are important and how they should be set.
