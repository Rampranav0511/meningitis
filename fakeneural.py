import math
import numpy as n
from sockfish import tr,label,valdata,data
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
    
    return actf





dimensions=[input_no] + outputsizes # this gives you the layer by layer input neurons
"""
first_act=forward_pass(input_layer,first_weights,first_bias)
# act is now the final output in the last layer , the last layer here has only one neuron btw
first_out=forward_pass(first_act,weights,bias) """
for i in range(0,len(dimensions)-1):
    
    
    weights=n.random.randn(dimensions[i+1],dimensions[i])# each thing is the weights matrix per layer
    bias=n.random.randn(dimensions[i+1],1) # bias aswelll
    weight_list.append(weights)
    bias_list.append(bias)
batchsize=64

total_epochs=100
best_val_loss=float('inf')
patience=5
counter=0
for e in range(total_epochs):

    for i in range(0,len(tr),batchsize):
                           
        batch=tr[i:i+batchsize]                             
        if len(batch)<2:                                     
            continue
        pos=n.hstack([x.reshape(-1,1) for x,y in batch])    # horizontally stacks input layers(coluwn wise adding)            
        evaltarget=n.array([y for x,y in batch]).reshape(1,-1)



         #tr is basically the array of tuples ( each tuple is each position , eval)
        

        layer_acts=[]
        layer_acts=[pos]


        # this is the forward pass layer
        act=pos
        for h in range(len(weight_list)):
            act=forward_pass(act,weight_list[h],bias_list[h])
            layer_acts.append(act)


        activation=layer_acts[-1]

        de=activation-evaltarget
        # backprop starts from here 
        for j in range(len(weight_list)-1,-1,-1):
            present_act=layer_acts[j+1]
            prev_act=layer_acts[j]

            wog=weight_list[j]
            nudge=de@prev_act.T
            biasnudge=de.sum(axis=1,keepdims=True)

            weight_list[j]=wog-step*nudge
            bias_list[j]-=step*biasnudge
            """we give the if condition later because for the first iteration
            the output error is basically final activation - target , later on it gets
            multiplied with the derivative of activation function of previous layers"""
            if j!=0:
                de=(wog.T@de)*(prev_act*(1-prev_act))
    # validation layer (forward pass of new examples but on updated weights in every epoch)
                
    valloss=0
    for position,target in valdata:
        act=position.reshape(-1,1)
        for i in range(len(weight_list)):

            act=weight_list[i]@act + bias_list[i]
            act=1/(1+n.exp(-act))
        valloss+=n.mean((act-target)**2)
    avg_valloss=valloss/len(valdata)


    #early stopping check
    if avg_valloss < best_val_loss:
        best_val_loss=avg_valloss
        counter=0
        dict={}

        for index,(weih,bias) in enumerate(zip(weight_list,bias_list)):
            dict[f'weih{index}']=weih
            dict[f'bias{index}']=bias

            """we basically ran a loop and saved all the weights 
            in the dictionary and then we are saving the dictionary as a .npz file using savez in numpy"""
        n.savez("neural network",*weight_list,*bias_list)    
    else:
        counter+=1
        if counter>=patience:
            break

    








    








            
    