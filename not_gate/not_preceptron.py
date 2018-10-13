import pandas as pd

#graph
#											_ Yes
# /*input*\		/*node*\				  /
#     X1 ----> (W1*X1 + b) ------> result-
#		  w1							  \ _ No
#


#defining weights

weight1 = -2

bias = 1

#definig test inputs
test_inputs = [0, 1]

#defining correct outputs
correct_outputs = [True, False]

#3defining the output array
outputs = []

#checking for each for
for test_input, correct_output in zip(test_inputs, correct_outputs):

	#calculating lenear combinaton
	linear_combination = test_input*weight1 + bias

	#checking output is +ve or not y-lable
	output = (linear_combination >= 0)

	#setting strin up
	is_correct_string = "Yes" if output == correct_output else "No"

	#pushing result in list
	outputs.append([test_input, linear_combination, output, is_correct_string])

#checking if there is no error
num_wrong = len([output[3] for output in outputs if output[3] == "No"])

#converting to table format
output_frame = pd.DataFrame(outputs, columns=["Input", "Linear_Combination", "Output", "Is_Correct"])

#printing result
if not num_wrong:

	print("\nNice! we got it right.\n")
else:
	print("\nSomewhere wrong {} keep trying".format(num_wrong))

print(output_frame.to_string(index = False))

#-------output is------

# Input  Linear_Combination  Output Is_Correct
#    0                   1    True        Yes
#    1                  -1   False        Yes

