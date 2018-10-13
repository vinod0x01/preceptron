import pandas as pd

#these are the wights and bias for the and preceptron
weight1 = 1
weight2 = 1
bias = -2

#test input for and preceptron
test_inputs = [(0,0),(0,1),(1,0),(1,1)]

#correct output for and preceptron
correct_outputs = [False, False, False, True]

#final output array for linear equation
outputs = []

#run linear equation based on weights for all inputs and bias
#check with the original one and store it in outputs yes if equal else No

#the graph look like this


"""
node simulation
	O\
	  ()-->
	O/
"""

# X1\
#.   \w1 								 _ Yes
#.    \									/
#.     (x1*w1 + x2*w2 + b)------> result
#.    / 								\
#    /w2 								 - No
# X2/
#

for test_input, correct_output in zip(test_inputs, correct_outputs):

	leaner_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias

	output = int(leaner_combination >= 0)

	is_correct_string = 'Yes' if output  == correct_output else 'No'

	outputs.append([test_input[0], test_input[1], leaner_combination, output, is_correct_string])

#check if any prediction are wrong or not
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])

#format output to table view
output_frame = pd.DataFrame(outputs,columns=['Input 1', 'Input 2', 'Leaner Combination', 'Activation Output', 'Is Correct'])

#print_ output
if not num_wrong:
	print("Nicee! you got it all correct.\n")
else:
	print("You got Wrong {} Keep trying!\n".format(num_wrong))

print(output_frame.to_string(index = False))


#----output----

# Input 1  Input 2  Leaner Combination  Activation Output Is Correct
#       0        0                  -2                  0        Yes
#       0        1                  -1                  0        Yes
#       1        0                  -1                  0        Yes
#       1        1                   0                  1        Yes

