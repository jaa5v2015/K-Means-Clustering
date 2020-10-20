#!/usr/bin/env python

import sys
import numpy as np
import math
from math import log

class Node():
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None
        self.catagory = None
        self.terminal = None
        self.classification = None

def getInfo(numClassifications,head, row, column, training_data, indices):
    info = 0.0
    prob = 0.0
    numClass = 0
    I = 0
    for i in range(numClassifications):
        numClass = 0
        for j in range(head,row):
            if(training_data[indices[j][column]][training_data.shape[1]-1] == i):
                numClass += 1
        prob = numClass / training_data.shape[0]
        if(numClass != 0):
            I = -(prob) * np.log2(prob)
        else:
            I = 0
        info += I
    return info


def calculate_split(I, classifications, numAttributes, column, training_data, indices):
    min_info = I
    for x in range(numAttributes-1):
        for y in range(column-1):
            if(training_data[indices[y][x]][x] != training_data[indices[y+1][x]][x]):
                E = (y+1) / column * getInfo(classifications,0, y+1, x, training_data, indices) + (column - (y+1)) / column *getInfo(classifications,y+1, column, x, training_data, indices)
                max_gain = I - E
                if(E < min_info):
                    min_info = E
                    calc_split = (training_data[indices[y][x]][x] + training_data[indices[y+1][x]][x]) / 2
                    index = x
    return calc_split, index

def get_newSize(column, catagory, value, training_data, indices):
    count1 = 0
    count2 = 0
    for j in range(column):
            if(training_data[indices[j][catagory]][catagory] < value):
                count1 += 1
            else:
                count2 += 1
    return count1, count2 
    
def build_node(training_data, indices,numClassifications):
    numAttributes = training_data.shape[1]
    column = training_data.shape[0]
    num_class = 0
    I = 0
    
    #Get the starting information
    I = getInfo(numClassifications, 0, column, 0, training_data, indices)
    #If terminal node
    if(I == 0):
        new_node = Node()
        new_node.terminal = True
        new_node.classification = training_data[indices[0][0]][numAttributes-1]
        new_node.catagory = None
        new_node.value = -1
        new_node.right = None
        new_node.left = None
        return new_node
    else:
        new_node = Node()
        new_node.terminal = False
        new_node.classification = -1
        new_node.value, new_node.catagory = calculate_split(I, numClassifications, numAttributes, column, training_data, indices)
        
        row_left , row_right = get_newSize(column, new_node.catagory, new_node.value, training_data, indices)
        
        right_childData = np.zeros([row_right, numAttributes])
        left_childData = np.zeros([row_left, numAttributes])
        
        left_count = 0 
        right_count = 0
        for j in range(column):
            if(training_data[indices[j][new_node.catagory]][new_node.catagory] < new_node.value):
                left_childData[left_count,:] = training_data[indices[j,:]][new_node.catagory]
                left_count += 1
            else:
                right_childData[right_count,:] = training_data[indices[j,:]][new_node.catagory]
                right_count += 1
        
        indices_right = np.argsort(right_childData, axis = 0)
        indices_left = np.argsort(left_childData, axis = 0)
        
    
        new_node.right = build_node(right_childData,indices_right,numClassifications)
        new_node.left = build_node(left_childData, indices_left,numClassifications)    
        return new_node
                 
def main():
    #Get data from command line
    file_training = sys.argv[1]
    file_testing = sys.argv[2]
    numAttributes = 0
    numClassifications = 1
    test_rows = 0
    correct = 0
    #open training data
    training_data = np.loadtxt(file_training)
    
    #open testing data
    testing_data = np.loadtxt(file_testing)
    
    
    if(len(training_data.shape) < 2):
        training_data = np.array([training_data])
        
    if(len(testing_data.shape)<2):
        testing_data = np.array([testing_data])
        
    indices = np.argsort(training_data,axis=0)
    
    numAtrributes = training_data.shape[1]
    for x in range(training_data.shape[0]-1):
        if(training_data[indices[x][numAttributes - 1]][numAttributes-1] != training_data[indices[x+1][numAttributes - 1]][numAttributes-1]):
            numClassifications += 1
            
    indices = np.argsort(training_data,axis=0)
    #Build decision tree
    root = build_node(training_data, indices,numClassifications)
    
    current = root
    c = True
    count = 0
    #Run test
    while(count < testing_data.shape[0]):
        c = True
        current = root
        while(c == True):
            if(current.terminal == True and current.classification != testing_data[[count][numAttributes-1]][numAttributes-1]):
                c = False
            elif(current.terminal == True and current.classification == testing_data[[count][numAttributes-1]][numAttributes-1]):
                correct += 1
                c = False
            elif(testing_data[count][current.catagory] < current.value and current.left != None):
                print("going to left node")
                current = current.left
            elif(testing_data[count][current.catagory] > current.value and current.right != None):
                print("going to right node")
                current = current.right
            else: 
                c = False
        count += 1
    print(correct)
        
if __name__ == "__main__":
    main()
    