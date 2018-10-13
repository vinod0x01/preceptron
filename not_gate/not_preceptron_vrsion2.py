import pandas as pd

#the graph look like this

# X1\
#.   \w1 								 _ Yes
#.    \									/
#.     (x1*w1 + x2*w2 + b)----->(result)
#.    / 								\
#    /w2 								 - No
# X2/
#

#defining the weights and bias

#since we are rejecting first input
weight1 = 0

weight2 = -2

bias = 1

#defining the test inputs
test_inputs = [(0,0),(0,1),(1,0),(1,1)]

#defining the correct otputs
correct_outputs = [True, False, True, False]

outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):

	#the result of equation
	linear_combination = test_input[0] * weight1 + test_input[1] * weight2 + bias

	#step function
	output = (linear_combination >= 0)

	#checking with correct output
	is_correct_string = "Yes" if output == correct_output else "No"

	#appnding to output list
	outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

#check the number of wrong predictions
num_wrong = len([output[4] for output in outputs if output[4] == "No"])

#convert to table format
output_data = pd.DataFrame(outputs, columns=(["Input1", "Input2", "Linear_combination", "Output", "IS_Correct"]))

if not num_wrong:

	print("!Nice all are correct.")
else:
	print("You got {} Wrong output! keep trying.".format(num_wrong))


print(output_data)

#----output will be----
# !Nice all are correct.
#    Input1  Input2  Linear_combination  Output IS_Correct
# 0       0       0                   1    True        Yes
# 1       0       1                  -1   False        Yes
# 2       1       0                   1    True        Yes
# 3       1       1                  -1   False        Yes

