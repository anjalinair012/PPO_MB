import re
import pdb


Next=[]
Reward=[]
State = []
Action=[]
Done = []
Time = []

def fileRead(filename):
	with open(filename) as myfile:
		line = myfile.readline()
		while line != '':
			state = list(map(float, line.split(',')))
			action = list(map(float, myfile.readline().split(',')))
			next_state = list(map(float, myfile.readline().split(',')))
			line = list(myfile.readline().split(','))
			reward = float(re.findall("\d+\.\d+", line[0])[0])
			done = bool("True") if myfile.readline()[0] == 'F' else bool("False")
			time = float(myfile.readline()[:-1].replace(",",""))
			#done = bool(line[2].strip())
			State.append(state)
			Action.append(action)
			Next.append(next_state)
			Reward.append(reward)
			Done.append(done)
			Time.append(time)
			line = myfile.readline()
	return State,Action,Next, Reward, Done, Time


if __name__ == "__main__":
	d,n = fileRead()
