Code for the implementation of Bachelor Project: Deep Reinforcement Learning for Physics-based Musculoskeletal Simulations of Healthy Subjects and Transfemoral Prosthetic Users during Normal Walking.
The code is largely inspired by the 7th-placed solution to the 2018 NIPS AI For Prosthetics competition: http://osim-rl.stanford.edu/docs/nips2018/. OpenAI's Baselines implementation of PPO serves as the basis for the learning algorithm. To speed up training, state trajectories at different walking speeds are included in the osim-rl/osim/data folder.


Running the code: 
Open terminal, enter the installed opensim environment with: source activate opensim_rl
Run: python run_command_file 
