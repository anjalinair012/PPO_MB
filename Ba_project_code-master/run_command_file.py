import time
import os
import subprocess, signal

#max_seconds = 216000
# Every 3 hours, a new process is started
max_seconds = 10800

tstart = time.time()
tstart2 = time.time()
last_timestep = 0
max_timesteps = 12000000

args = ["mpirun", "-np", "4", "python", "main.py", "1", "1"]

#pid = os.fork()
proc = subprocess.Popen(args)

# Runs for max_timesteps
while(last_timestep < max_timesteps):
  time.sleep(10)
  if(time.time() - tstart > max_seconds):
     subprocess.Popen.kill(proc)
     # Get iterations and timesteps from file
     with open('iterations.txt', 'r') as f:
          lines = f.read().splitlines()
          # Get the last line as the last stored iteration
          last_iter = int(lines[-1])
     with open('timesteps.txt', 'r') as g:
          lines = g.read().splitlines()
          # Get the last line as the last stored time step
          last_timestep = int(lines[-1])

     tstart = time.time()
     args = ["mpirun", "-np", "4", "python", "main.py", "1", "1"]
     proc = subprocess.Popen(args)

subprocess.Popen.kill(proc)








