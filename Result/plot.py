from numpy import genfromtxt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
per_data=genfromtxt('128.txt',delimiter=',')
X=[]
Y=[]
red_patch = mpatches.Patch(color='orange', label='SVMResults')

bluepatch = mpatches.Patch(color='blue', label='AlgorithmResults')
plt.legend(handles=[red_patch,bluepatch])
for i in range(len(per_data)):
	X.append(per_data[i][0])
	Y.append(per_data[i][1])


plt.xlabel ('No. of Labeled data')
plt.ylabel ('Accuracy')
# plt.title('Results')
plt.plot(X,Y)

per_data=genfromtxt('svmans.txt',delimiter=',')
X=[]
Y=[]
for i in range(len(per_data)):
	X.append(per_data[i][0])
	Y.append(per_data[i][1])

plt.xlabel ('No. of Labeled data     (c)')
plt.ylabel ('Accuracy')
# plt.title('my test result')
plt.plot(X,Y)
plt.show()
