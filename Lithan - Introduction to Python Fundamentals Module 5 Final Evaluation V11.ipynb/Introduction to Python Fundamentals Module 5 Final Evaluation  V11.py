#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import string

# Download Data File
get_ipython().system('curl https://raw.githubusercontent.com/MicrosoftLearning/intropython/master/elements1_20.txt  -o elements1_20.txt')
    
# Open file to read amended list
elements_file = open('elements1_20.txt', 'r+')

# Determine length of list from file
# Initialize Counter
num_lines = 0
for line in elements_file:
    num_lines += 1
# print(num_lines) 

# Setup seek to count from 1st line
elements_file.seek(0)

# Setup up loop to read file and strip whitespaces
# Initialize counter
i = 1
elements_contents = []
for i in range(0, num_lines):
    # Store values from file into variable
    elements_contents.append(elements_file.readline().strip() + ",")
    i += 1    
print("Elements:")   
print(elements_contents)
elements_file.close

try:

    def get_names(elements_contents):

        # Input elements
        print("\nList any 5 of the first 20 elements in the Periodic Table. No digits. No \"Enter\". \n")
        # Counter
        cnt = 0
        bb = 0
        # Total number of entries
        leng = 5

        # Create empty lists
        element_list = []
        element_list2 = []
        not_found_list = []
        all_value_list = []
        all_value_list2 = []

        while cnt  < leng :

            # Input element
            element_input = ""
            a = 1
            while element_input.isdigit() or element_input == "" or a == 0:
                element_input = input(f"Enter the name of an element ({cnt + 1} of {leng} entries) or \"Q\" to quit: ")
    #             cnt += 1
                # Test for punctuations.
                for i in element_input:                
                    if (i in string.punctuation) or (i.isdigit()):
                        print(f"{element_input} is an invalid input. Please ONLY input a word or \"Q\" to quit.")
                        a = 1
    #                     cnt -= 1
                        element_input = ""
                        break  

            # Quit program
            if element_input.upper() == "Q":
    #             bb = 1
                print("\nProgram Terminated.")
                sys.exit()

            # Add "," to the end of input
            element_input += ","

            # Check for duplicate inputs
            # Fill values
            all_value_list.append(element_input)
            if element_input in all_value_list[:-1]:
                element_input = element_input.replace(",", "!")
                print(f"Duplicate value found for {element_input.title()} Please enter new input!")
                cnt -= 1

            # Initialize list for values not found in file
            non_elements = []
            # Loop through values to find elements not in file
            for m in all_value_list:
                if (m.title() not in elements_contents) and (m.title() not in non_elements):
                    non_elements.append(m.title())                

            # Check for element
            for e in elements_contents:
                # Check for values found in file        
                if e.lower() == element_input.lower():
                    element_list.append(e)
                    element_len = len(element_list)
                    element_list2 = []
                    # Check for duplicates
                    for i in element_list:
                        if i not in element_list2:
                            element_list2.append(i)

            # Update counter
            cnt += 1

        # Return values
        return all_value_list, element_list, element_list2, leng, non_elements, bb

    # Call Function
    all_value_list, element_list, element_list2, leng, non_elements, bb = get_names(elements_contents)

    # if bb == 1:
    #     os._exit(1)

    # print("EList2", element_list2) 
    leng_nfl = len(non_elements)  
    # print("Length NFL", leng_nfl)
    # print("el2 len", len(element_list2))
    # Create index value to slice list 
    # slice_val=leng – 1 – leng_nfl
    # print("Slice Value", slice_val)  

    # Sort lists
    element_list2.sort()
    non_elements.sort()

    # Remove from last word "," and replace it with "."
    if len(element_list2) > 0:
        element_list2[-1] = element_list2[-1].replace(",", '.')
    if len(non_elements) > 0:
        non_elements[-1] = non_elements[-1].replace(",", '.')

    # Print output
    if len(non_elements) == 0:
        print("\nFound:", *element_list2, end = " ")
        print("\nAll values found in Periodic Table.")
    elif len(element_list2) == 0:
        print("\nNot Found: ", *non_elements, end = " ")
        print("\nNo values found in Periodic Table.")
    else:
        print("\nFound:", *element_list2, end = " ")
        print("\nNot Found: ", *non_elements, end = " ")


    # Print out correct elements 
    # for j in element_list:
    # print("\nLista", element_list)
    # print("nfl2", not_found_list)
    # print("\nel3", element_list3)
    # print("\nFound:", *element_list2 , end = " ") 
    # print("\nNot Found: ", *not_found_list, end = " ")

    # Calculate %
    # Length of list
    len_list = leng 
    # Length of values found
    values_found = len(element_list2)
    # Length of values not found
    values_not_found = len(non_elements)
    # formula is: (values_found/values_not_found) * 100
    correct_percentage = (values_found/len_list) * 100
    print(f"\n{correct_percentage}% of values found in Periodic Table.")                 

except:
    pass


# In[ ]:





# In[ ]:




