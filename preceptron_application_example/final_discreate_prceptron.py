import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import csv

#using grid to plot the graphs
style.use('ggplot')
cwd = str(os.getcwd())
x = [[],[]]
y = []
W = []
b = 0

#data file (path of student_data.csv)
data_file_path = "student_data.csv"

#function to read the csv data from the file
def read_data():

	#reading from file
	with open(data_file_path) as data_file:

		#extracting data from csv file 
		data = csv.reader(data_file, delimiter=',')

		for row in data:

			#assigning row values
			#(test, grade, result)
			#x = [[tests],[grades]]
			#y = [results]
			x[0].append(float(row[0]))
			x[1].append(float(row[1]))
			y.append(int(row[2]))

#function to draw graph at the end
def draw_graph(line,itr):

	#assining the label and data
	plt.xlabel('Test Marks')
	plt.ylabel('Grades')
	plt.title('Student data')
	
	#defining the limits of x-axis and y-axis
	plt.xlim(min(x[0])-0.5,max(x[0])+0.5)
	plt.ylim(min(x[0])-0.5,max(x[1])+0.5)

	#first draw the points
	for i in range(0,len(y)):

		#check for result
		if y[i] == 0:
			#blue point
			plt.plot(x[0][i], x[1][i], 'bo')

		elif y[i] == 1:
			#red point
			plt.plot(x[0][i], x[1][i], 'ro')
	#draw the line
	plt.plot(line[0], line[1], label='itaration - '+itr,color='g')
	plt.legend()
	plt.show()

#defining the preceptron

#the graph for discreate preceptron
#
# X1_
#	 \
#	  \	 (step func) __ YES__
#      \     __     /	     \
# X2----( __|   )---          > y^
#	   /			\__ NO___/
#     /
#	 /
#	/
# b

#defining the step function
#function to check the preceptron

def stepFunction(t):

	if t >= 0:
		return 1
	return 0

#function to calc WX + B predicting the step value
def prediction(X, W, b):

	#to calculate WX+b
	#return 1 if +ve point in -ve area
	#return 0 if -ve point in +ve area
	return stepFunction((np.matmul(X,W)+b)[0])

#functon for moving the line
def preceptronStep(X, y, W, b, learn_rate = 0.01):

	for i in range(0, len(X)):

		y_hat = prediction(X[i], W, b)

		if y[i]-y_hat == 1:
			'''
				now the point has to be accepted, but according to boundry line
				it is rejected so change weights according to move line towards blue point	
			'''
			#now the point has to be accepted, but according to boundry line
			#it is rejected so change weights 
			# +ve point in -ve area then add 
			# w1 = w1 + (x1 * alpha)
			# w2 = w2 + (x2 * alpha)
			# bias = bias + alpha
			# where alpha is a learning rate 

			W[0] += (X[i][0]*learn_rate)
			W[1] += (X[i][1]*learn_rate)
			b += learn_rate

		elif y[i]-y_hat == -1:

			'''
				now the point has to be rejected, but according to boundry line
				it is accepting so change weights according to move line towards redpoint point	
			'''
			# -ve point in +ve area then subtract
			# w1 = w1 - (x1 * alpha)
			# w2 = w2 - (x2 * alpha)
			# bias = bias - alpha
			# where alpha is a learning rate 

			W[0] -= (X[i][0]*learn_rate)
			W[1] -= (X[i][1]*learn_rate)
			b -= learn_rate

	return W, b

#function to find the equation of boundary lines
def trainPreceptronAgorithm(X, y, learn_rate=0.01, num_epoches=100):

	global W,b
	#finding max and min values
	x_min, x_max = min(X.T[0]), max(X.T[0])
	y_min, y_max = min(X.T[1]), max(X.T[1])

	#initially start with the random weights
	W = np.array(np.random.rand(2,1)) 

	#also with random bias
	b = np.random.rand(1)[0] + x_max

	#initialising the boundry lines which are solutions get ploted on the graph
	boundary_lines = []

	#training the preceptron to get correct graph 
	for i in range(0,num_epoches):
		#training the preceptron
		W, b = preceptronStep(X, y, W, b, learn_rate)

		#appending boundry lines
		'''
			from eqyation w1x1 +w2x2 + b = 0 (WX + b = 0)

			x1 = -b/w1

			x2 = -b/w2
		'''
		boundary_lines.append([-b/W[0],-b/W[1]])

	return boundary_lines

if __name__ == '__main__':

	#read the data
	read_data()
	
	#converting the red data to np array matrix
	X = np.array([x[0], x[1]])
	
	#training the preceptron
	ans = trainPreceptronAgorithm(X.T, y)

	'''
		drawing the graph for all the itarations
		from eqyation w1x1 +w2x2 + b = 0 (WX + b = 0)
		the points of boundry line are (x1,0) and (0, x2)
		  |
	(0,x2)|\
		  | \
		  |  \
		  |___\______________
		      (x1,0)
	'''
	for i in range(0,len(ans)):
		draw_graph([[ans[i][0][0], 0], [0, ans[i][1][0]]], str(i+1))
	#printing final weights and bias
	print("final weights and bias are : ")
	print("W1 = {}\n W2 = {}\n b = {}".format(-W[0][0], -W[1][0], -b))