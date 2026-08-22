import math
import numpy as n
input_no=int(input("enter input layer neuron number"))
output_no=int(input("enter first hidden layer neuron number"))
# pass x , because its the input to each game board neurons
first_weights=n.random.randn(output_no,input_no)

input_layer=n.random.randn(input_no,1)
step=0.01
first_bias=n.random.randn(output_no,1)
layer_acts=[]
weight_list=[]
bias_list=[]
target=0
# x=input_layer@first_weights + first_bias
x=[]
# first_act=1/(1+math.exp(-x))
outputsizes=[]
act=n.random.randn(input_no,1)
for i in range(3): # this indicates four( 3 hidden layers and a input layer) layers of the network , basically each layer contains how many neurons and stuff+
    outputsizes.append(int(input()))


def forward_pass(inp,weig,bia):
    act=weig@inp + bia
    actf=1/(1+n.exp(-act))
    layer_acts.append(actf)
    weight_list.append(weig)
    bias_list.append(bia)
    return actf






"""
first_act=forward_pass(input_layer,first_weights,first_bias)
# act is now the final output in the last layer , the last layer here has only one neuron btw
first_out=forward_pass(first_act,weights,bias) """
for i in range(0,len(outputsizes)-1):
    
    
    weights=n.random.randn(outputsizes[i+1],outputsizes[i])# each thing is the weights matrix per layer
    bias=n.random.randn(outputsizes[i+1],1) # bias aswelll
    weight_list.append(weights)
    bias_list.append(bias)
    
total_epochs=100
best_val_loss=float('inf')
patience=5
counter=0
for e in range(total_epochs):



    for pos,evaltarget in data:

        layer_acts=[]
        layer_acts.append(pos)


        # this is the forward pass layer
        act=pos
        for h in range(len(weight_list)):
            act=forward_pass(act,weight_list[h],bias_list[h])
        


        activation=layer_acts[-1]

        de=activation-evaltarget
        # backprop starts from here 
        for j in range(len(weight_list)-1,-1,-1):
            present_act=layer_acts[j+1]
            prev_act=layer_acts[j]


            nudge=de@prev_act.T
            biasnudge=de

            weight_list[j]-=step*nudge
            bias_list[j]-=step*biasnudge
            """we give the if condition later because for the first iteration
            the output error is basically final activation - target , later on it gets
            multiplied with the derivative of activation function of previous layers"""
            if j!=0:
                de=(weight_list[j].T@de)*(present_act*(1-present_act))
    # validation layer (forward pass of new examples but on updated weights in every epoch)
                
    valloss=0
    for position,target in valdata:
        act=position
        for i in range(len(weight_list)):
            act=weight_list[i]@act + bias_list[i]
            act=1/(1+n.exp(-act))
        valloss+=n.mean((act-target)**2)
    avg_valloss=valloss/len(valdata)


    #early stopping check
    if avg_valloss < best_val_loss:
        best_val_loss=avg_valloss
        counter=0
    else:
        counter+=1
        if counter>=patience:
            break

    








    








            
    