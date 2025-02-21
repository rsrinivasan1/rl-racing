# RL with PPO on the CarRacing environment

This project is an extension of my prior work on the Cartpole environment. My main goal here is to gain a better understanding of reward mechanisms in proximal policy optimization, and to discover the limits of my PPO implementation in PyTorch with limited hyperparameter tuning. This is still a work-in-progress (as of Feb 2025), and I hope to eventually "solve" this environment (achieving a mean score of 900 over 100 trials). This is a tough goal and will require some further work.

## Method

My main network architecture consists of Actor and Critic networks that share convolutional layers, taking in the input frames from the environment. I stack the four most recent frames together, first preprocessing the frames by turning the track black and compressing the image to black and white.

<img width="856" alt="Screenshot 2025-02-21 at 2 36 27 AM" src="https://github.com/user-attachments/assets/7da87161-859b-4253-9403-061b7a32562e" />

For the action space, I chose to model the normally continuous settings for turning and acceleration by mapping them to 5 discrete actions:

- Turn left
- Turn right
- Accelerate
- Brake
- Do nothing

Each action taken by the agent is repeated for 4 frames of 96x96 pixels to make the agent's actions a bit smoother and reduce the computational load. I find that this configuration does not limit the model's ability to learn effectively.

Currently, I am experimenting with learning rate annealing, although I'm not sure it makes a huge difference in earlier stages of training. I'm also testing different reward systems — for instance penalizing the agent for staying on the grass too long — and looking into whether limiting the agent's max acceleration is useful.

## Results

### Best run so far:

https://github.com/user-attachments/assets/65d0535a-9e2d-46be-a916-2d0491016a6c

### A faulty but very speedy run (with grass penalty):

https://github.com/user-attachments/assets/dfb50112-20b8-45f0-8edf-640ce7143a05

## Next steps

I'm looking forward to eventually cracking this environment, and obtaining a speedy and highly accurate model. I'd like to see how far I can get with my current standard PPO implementation, with some mild reward tuning, but I'm also open to the possibility of reducing the image input size, giving the model access to speed information, and even splitting the policy network into two (one for thrusting, one for steering, as done here: https://github.com/Ceudan/Car-Racing).

My goal with this project is to build an agent that could consistently outperform a reasonably capable human.

## Resources
I must credit a variety of resources that were extremely helpful, namely the following:

- https://notanymike.github.io/Solving-CarRacing/
- https://github.com/xtma/pytorch_car_caring
- https://github.com/LucaZheng/CarRacingV2_PPO
- https://github.com/Ceudan/Car-Racing
- https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases

These proved extremely helpful in determining how to structure my networks and tune certain parameters that would have been difficult to uncover otherwise. This includes obtaining an efficient equation for Generalized Advantage Estimation and adding an entropy term to my loss function, which greatly helped when training.
