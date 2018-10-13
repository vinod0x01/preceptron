import pandas as pd


#the multi layer graph looks like

#	Input-1------(AND)----(NOT)__
#		   \.   / 			     \
#			\  /				  \
#			 \/					   \
#			 /\					   (And)-------> XOR
#			/  \				  /
#		   /.   \				 /
#	Input-2 ----(OR)____________/
#

def and_it(input1, input2):

	weight1 = 1
	weight2 = 1
	bias = -2

	res = liniar_combination(input1, input2, weight1, weight2, bias)

	output = (res >= 0)

	return output

def or_it(input1, input2):

	weight1 = 1
	weight2 = 1
	bias = -1

	res = liniar_combination(input1, input2, weight1, weight2, bias)

	output = (res >= 0)

	return output

def not_it(input1):

	return not(input1)

def liniar_combination(x1, x2, w1, w2, b):

	res = x1*w1 + x2*w2 + b

	return (res)

#defining the test inputs and correct outputs

test_inputs = [(0,0), (0,1), (1,0), (1,1)]

correct_outputs = [False, True, True, False]

outputs = []

for test_input, correct_output in zip(test_inputs,correct_outputs):

	#calculate output by different layer
	output = and_it( int( not_it( and_it(test_input[0], test_input[1]))), int( or_it(test_input[0], test_input[1])))

	#check it is correct or not
	is_correct = "Yes" if output == correct_output else "No"

	#append to output
	outputs.append([test_input[0], test_input[1], output, is_correct])

#check for wrong predictions
num_wrong = len([output[3] for output in outputs if output[3] == "No"])

#convert to table format
output_data = pd.DataFrame(outputs, columns=(["Input-1", "Input-2", "Output", "IS_Correct"]))

#print result
if not num_wrong:
	print("Nice we got all right.")
else:
	print("we got {} wrong outputs!".format(num_wrong))

print(output_data)

#----output will be-------

# Nice we got all right.
#    Input-1  Input-2  Output IS_Correct
# 0        0        0   False        Yes
# 1        0        1    True        Yes
# 2        1        0    True        Yes
# 3        1        1   False        Yes

