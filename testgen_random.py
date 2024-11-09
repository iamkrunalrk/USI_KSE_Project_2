import os
import numpy as np
import sys
import instrumentor as inst  # Import the custom 'instrumentor' module
import random
from deap import creator, base, tools, algorithms  # Import DEAP (Distributed Evolutionary Algorithms in Python) for evolutionary algorithms
import shutil
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon  # Import Wilcoxon test for statistical analysis

# ------------------------ HYPERPARAMETERS ------------------- #

# Constants for generating random strings and integers
MIN_INT = -1_000  # Minimum value for random integers
MAX_INT = 1_000   # Maximum value for random integers
MAX_STRING_LENGTH = 10  # Maximum length of a random string
POOL_SIZE = 1_000  # Size of the pool of random variables

# Number of copies of test archive to generate
COPIES = 10

# ------------------------ Functions ------------------------ #

# Function to get the list of file paths in the "parsed" folder
def parsed_paths():
    list_of_files = []  # List to store file paths
    files_in_benchmark = "parsed"  # Directory containing the parsed files
    
    # Loop through files in the "parsed" directory
    for filename in os.listdir(files_in_benchmark):
        file_path = os.path.join(files_in_benchmark, filename)
        
        # Add file to the list if it's a regular file (not a directory)
        if os.path.isfile(file_path):
            list_of_files.append(file_path)
    
    return list_of_files

# Function to read the content of each file in the parsed folder
def parsed_codes(list_of_files):
    list_of_codes = []  # List to store code contents
    dict_codes = {file: [] for file in list_of_files}  # Dictionary to store file paths and their corresponding code
    
    # Read code from each file in the list_of_files
    for file_path in list_of_files:
        with open(file_path, 'r') as file:
            code = file.read()  # Read the content of the file
            list_of_codes.append(code)  # Add the code to the list
            dict_codes[file_path] = code  # Store the code in the dictionary
    
    return list_of_codes, dict_codes

# Function to extract function signatures from the code in each parsed file
def parsed_signatures(my_dict):
    for idx in my_dict:  # Loop through the files in the dictionary
        lines = my_dict[idx].split("\n")  # Split code into lines
        signature = ""  # Initialize an empty string to hold function signatures
        
        # Iterate over lines to find function definitions
        for line in lines:
            if line.startswith("def"):  # Check if the line defines a function
                if signature != "":  # If we already have a signature, add a new one
                    signature += "\n new_signature " + line
                else:
                    signature += line  # Otherwise, start the signature with the line
            if signature != "":  # Once a signature is found, update the file in the dictionary
                my_dict[idx] = signature
    
    return my_dict

# Function to get the types of parameters from each function signature
def get_types_of_parameters(my_dict):
    for idx in my_dict:  # Loop through the files in the dictionary
        signatures = my_dict[idx]  # Get the list of signatures for this file
        types = []  # List to store types for each signature
        
        # Loop through the signatures of each function
        for signature in signatures:
            # Extract the parameter part of the function signature (between parentheses)
            start = signature.index("(") + 1
            end = signature.index(")")
            params = signature[start:end]
            
            # Split parameters by commas
            split_by_comma = params.split(",")
            
            tipos = []  # List to store types for the parameters
            for arg in split_by_comma:
                # Extract the type from the argument (split by colon)
                arg_type = arg.split(":")[-1]
                tipos.append(arg_type)  # Append the type to the list
            
            types.append((signature, tipos))  # Store the signature and its parameter types
        
        my_dict[idx] = types  # Update the dictionary with the extracted types
    
    return my_dict

# ------------------------ Random Variables ------------------------ #

# Function to generate a random string of random length
def init_str_variable():
    length = np.random.randint(0, MAX_STRING_LENGTH)  # Randomly choose a length for the string
    sentence = ""  # Initialize an empty string
    
    # Generate a random string of lowercase letters
    for _ in range(length):
        sentence += chr(np.random.randint(97, 122))  # Randomly choose a lowercase letter
    
    return sentence  # Return the random string

# Function to generate a random integer within the specified range
def init_int_variable():
    return np.random.randint(MIN_INT, MAX_INT)  # Return a random integer between MIN_INT and MAX_INT

# Function to generate a random key-value pair (string, integer)
def init_key_value_pairs():
    return (init_str_variable(), init_int_variable())  # Generate a random string and integer pair

# Function to generate a pool of random strings
def single_string():
    sentences = []  # List to store the random strings
    for _ in range(POOL_SIZE):
        sentences.append(init_str_variable())  # Add a random string to the list
    return sentences  # Return the pool of random strings

# Function to generate a pool of random integers
def single_int():
    int_variables = []  # List to store the random integers
    for _ in range(POOL_SIZE):
        int_variables.append(init_int_variable())  # Add a random integer to the list
    return int_variables  # Return the pool of random integers

# Function to generate a pool of random pairs of strings
def double_string():
    sentences = []  # List to store the random pairs of strings
    for _ in range(POOL_SIZE):
        sentences.append((init_str_variable(), init_str_variable()))  # Add a pair of random strings
    return sentences  # Return the pool of random string pairs

# Function to generate a pool of random pairs of integers
def double_int():
    int_variables = []  # List to store the random pairs of integers
    for _ in range(POOL_SIZE):
        int_variables.append((init_int_variable(), init_int_variable()))  # Add a pair of random integers
    return int_variables  # Return the pool of random integer pairs

# Function to generate a pool of random key-value pairs (string, integer)
def string_int():
    key_value_pairs = []  # List to store random key-value pairs
    for _ in range(POOL_SIZE):
        key_value_pairs.append(init_key_value_pairs())  # Add a random key-value pair
    return key_value_pairs  # Return the pool of random key-value pairs

# Function to generate a pool of random triplets of integers
def triple_int():
    int_variables = []  # List to store random triplets of integers
    for _ in range(POOL_SIZE):
        int_variables.append(
            (init_int_variable(), init_int_variable(), init_int_variable())  # Add a triplet of random integers
        )
    return int_variables  # Return the pool of random integer triplets


# ------------------------ Create Pools ------------------------ #

# Function to create a pool of variables based on the provided element types
def create_pool(elem):
    # Check the number of elements in the tuple
    if len(elem) == 1:
        x = elem[0]  # Extract the first element
        
        # If the element is "str", generate a pool of random strings
        # Otherwise, generate a pool of random integers
        if x == "str":
            variables = single_string()
        else:
            variables = single_int()
    
    # If there are two elements in the tuple
    elif len(elem) == 2:
        x, y = elem[0], elem[1]  # Extract both elements
        
        # If both are strings, generate a pool of pairs of random strings
        # If the first is a string and the second is an integer, generate a pool of key-value pairs (string, int)
        # Otherwise, generate a pool of pairs of random integers
        if x == "str" and y == "str":
            variables = double_string()
        elif x == "str" and y == "int":
            variables = string_int()
        else:
            variables = double_int()
    
    # If there are more than two elements, generate a pool of triples of random integers
    else:
        variables = triple_int()
    
    return variables

# ------------------------ Mutation ------------------------ #

# Function to mutate a single string element
def mutate_single_string(elem):
    # If the string is empty, no mutation is needed
    if len(elem) == 0:
        return elem
    else:
        # Randomly pick an index in the string and change the character at that position to a new random lowercase letter
        idx = np.random.randint(0, len(elem))  # Choose a random index
        return elem[:idx] + chr(np.random.randint(97, 122)) + elem[idx+1:]  # Replace the character

# Function to mutate a single integer element by generating a new random integer
def mutate_single_int():
    return init_int_variable()

# Function to mutate a key-value pair (string key and integer value)
def mutate_tuple_key_value(key, value):
    # If the key is empty, generate a new random integer as the value
    if len(key) == 0:
        new_value = init_int_variable()
        return (key, new_value)
    
    # Randomly select an index to mutate
    idx = np.random.randint(len(key))
    
    # Mutate either the key or the value
    if np.random.uniform() < 0.5:
        # Mutate only the key (replace a character at the selected index)
        new_key = key[:idx] + chr(np.random.randint(97, 122)) + key[idx+1:]
        return (new_key, value)
    else:
        # Mutate only the value (generate a new random integer)
        new_value = init_int_variable()
        return (key, new_value)

# Function to mutate a tuple containing two strings
def mutate_tuple_str_str(str1, str2):
    # If both strings are empty, no mutation is needed
    if len(str1) == 0 and len(str2) == 0:
        return str1, str2
    
    # If one string is empty, mutate the other string
    if len(str1) == 0:
        idx2 = np.random.randint(0, len(str2))  # Select a random index in the second string
        new_str2 = str2[:idx2] + chr(np.random.randint(97, 122)) + str2[idx2+1:]  # Mutate the second string
        return (str1, new_str2)
    
    if len(str2) == 0:
        idx1 = np.random.randint(0, len(str1))  # Select a random index in the first string
        new_str1 = str1[:idx1] + chr(np.random.randint(97, 122)) + str1[idx1+1:]  # Mutate the first string
        return (new_str1, str2)
    
    # If both strings are non-empty, mutate one or both randomly
    idx1 = np.random.randint(0, len(str1))
    idx2 = np.random.randint(0, len(str2))
    
    if np.random.uniform() < 1/3:
        # Mutate only the first string
        new_str1 = str1[:idx1] + chr(np.random.randint(97, 122)) + str1[idx1+1:]
        return (new_str1, str2)
    elif np.random.uniform() < 1/3:
        # Mutate only the second string
        new_str2 = str2[:idx2] + chr(np.random.randint(97, 122)) + str2[idx2+1:]
        return (str1, new_str2)
    else:
        # Mutate both strings
        new_str1 = str1[:idx1] + chr(np.random.randint(97, 122)) + str1[idx1+1:]
        new_str2 = str2[:idx2] + chr(np.random.randint(97, 122)) + str2[idx2+1:]
        return (new_str1, new_str2)

# Function to mutate a tuple of two integers
def mutate_tuple_int_int(int1, int2):
    # Randomly decide whether to mutate the first integer, the second integer, or both
    if np.random.uniform() < 1/3:
        # Mutate the first integer
        new_int1 = init_int_variable()
        return (new_int1, int2)
    elif np.random.uniform() < 1/3:
        # Mutate the second integer
        new_int2 = init_int_variable()
        return (int1, new_int2)
    else:
        # Mutate both integers
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        return (new_int1, new_int2)

# Function to mutate a tuple of three integers
def mutate_tuple_int_int_int(int1, int2, int3):
    # Randomly select one of six mutation cases
    case = np.random.randint(1, 7)
    
    # Perform the mutation based on the randomly selected case
    if case == 1:
        new_int1 = init_int_variable()
        return (new_int1, int2, int3)
    elif case == 2:
        new_int2 = init_int_variable()
        return (int1, new_int2, int3)
    elif case == 3:
        new_int3 = init_int_variable()
        return (int1, int2, new_int3)
    elif case == 4:
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        return (new_int1, new_int2, int3)
    elif case == 5:
        new_int1 = init_int_variable()
        new_int3 = init_int_variable()
        return (new_int1, int2, new_int3)
    elif case == 6:
        new_int2 = init_int_variable()
        new_int3 = init_int_variable()
        return (int1, new_int2, new_int3)
    else:
        new_int1 = init_int_variable()
        new_int2 = init_int_variable()
        new_int3 = init_int_variable()
        return (new_int1, new_int2, new_int3)

# Function to perform mutation based on the element type
def mutation(elem):
    # If the element is a string, mutate the string
    if isinstance(elem, str):
        return mutate_single_string(elem)
    
    # If the element is an integer, mutate the integer
    elif isinstance(elem, int):
        return mutate_single_int()
    
    else:
        # If the element is a tuple, determine its type and mutate accordingly
        a, b = elem[0], elem[1]
        
        if isinstance(a, str) and isinstance(b, int):
            return mutate_tuple_key_value(a, b)
        
        elif isinstance(a, str) and isinstance(b, str):
            return mutate_tuple_str_str(a, b)
        
        # Handle integer tuples (either two integers or three integers)
        if len(elem) == 2:
            return mutate_tuple_int_int(a, b)
        else:
            return mutate_tuple_int_int_int(a, b, elem[2])


# ------------------------ Crossover ------------------------ #

# Function to perform crossover between two individuals (e.g., strings or integers)
def crossover(individual1, individual2):
    # Case 1: If both individuals are integers, swap their values
    if isinstance(individual1, int) and isinstance(individual2, int):
        return individual2, individual1
    
    # Case 2: If either individual is too short (length <= 1), no crossover occurs
    if len(individual1) <= 1 or len(individual2) <= 1:
        return individual1, individual2
    
    # Case 3: If both individuals have longer lengths, perform crossover
    else:
        # Extract the first element (head) and the rest (tail) from individual 1
        head1 = individual1[0]
        tail1 = individual1[1]
        
        # Case 4: If the head of individual 1 is an integer, handle accordingly
        if isinstance(head1, int):
            head2 = individual2[0]  # Extract head from individual 2
            tail2 = individual2[1]  # Extract tail from individual 2
            
            # Case 4.1: If both individuals have a third element, preserve it while swapping heads and tails
            if len(individual1) == 3 and len(individual2) == 3:
                offspring1 = (head1, tail2, individual1[2])  # First offspring
                offspring2 = (head2, tail1, individual2[2])  # Second offspring
                return offspring1, offspring2
            
            # Case 4.2: If there's no third element, swap only the heads and tails
            else:
                offspring1 = (head1, tail2)  # First offspring
                offspring2 = (head2, tail1)  # Second offspring
                return offspring1, offspring2
        
        # Case 5: If the head of individual 1 is a string and the tail is an integer
        elif isinstance(head1, str) and isinstance(tail1, int):
            head2 = individual2[0]  # Extract head from individual 2
            tail2 = individual2[1]  # Extract tail from individual 2
            
            # If either head is empty, no crossover is done
            if not head1 or not head2:
                return individual1, individual2
            
            # Perform crossover at a random position in the string heads
            pos = np.random.randint(0, min(len(head1), len(head2)))
            
            # Create new offspring by crossing over string heads, while keeping integer tails intact
            offspring1 = head1[:pos] + head2[pos:]
            offspring2 = head2[:pos] + head1[pos:]
            
            # Return offspring with original tails
            individual1 = (offspring1, tail1)
            individual2 = (offspring2, tail2)
            return individual1, individual2
        
        # Case 6: If both individuals are strings (of any length)
        else:
            # If either individual is empty, no crossover occurs
            if not individual1 or not individual2:
                return individual1, individual2
            
            # Perform crossover at a random position in the string
            pos = np.random.randint(0, min(len(individual1), len(individual2)))
            
            # Create offspring by combining the heads and tails of the two individuals
            offspring1 = individual1[:pos] + individual2[pos:]
            offspring2 = individual2[:pos] + individual1[pos:]
            
            return offspring1, offspring2


# ------------------------ Testing ------------------------ #

# This function seems to test various function calls dynamically
# (It is marked as "NOT USED ANYMORE", so we'll leave it as is for reference)
def testing_function(def_dict, dict_codes):
    for file in def_dict:
        print(file)
        file_path = file
        function_names = def_dict[file][::2]  # Extract function names
        variable_types = def_dict[file][1::2]  # Extract variable types
        print(function_names)
        print(variable_types)

        for idx, f in enumerate(function_names):
            args = create_pool(variable_types[idx])[:1][0]  # Create argument pool
            print(args)
            print(type(args))

            try:
                current_module = sys.modules[__name__]
                code = compile(dict_codes[file_path], filename="<string>", mode="exec")
                exec(code, current_module.__dict__)  # Execute the function code dynamically
                dict_distance_true, dict_distance_false = inst.get_distance_dicts()
                print(f"true distance {dict_distance_true} and {dict_distance_false}")

                # Call the function dynamically with the arguments
                if isinstance(args, tuple):
                    x = globals()[f](*args)
                else:
                    x = globals()[f](args)
                    
                print(f"x is {x}")
                print(f"true distance {dict_distance_true} and {dict_distance_false}")
                
                dict_distance_true.clear()
                dict_distance_false.clear()
            
            except AssertionError as e:
                print(f"AssertionError {e} for function {f}")
                pass
            except BaseException as e:
                print(f"BaseError {e} for function {f}")
                raise e
        
        print("--------------------------------------------------")


# ------------------------ FUZZY ------------------------ #

# Function to execute a given function with arguments and return the result along with distance dictionaries
def execute_function(file, f, args):
    # Retrieve the current module for execution
    current_module = sys.modules[__name__]
    
    # Compile the code and execute it within the module's namespace
    code = compile(file, filename="<string>", mode="exec")
    exec(code, current_module.__dict__)
    
    # Get the true and false distance dictionaries
    dict_distance_true, dict_distance_false = inst.get_distance_dicts()
    
    # Call the specified function with the arguments
    if isinstance(args, tuple):
        y = globals()[f](*args)
    else:
        y = globals()[f](args)
    
    # If the result is a string, escape special characters
    if isinstance(y, str) and len(y) >= 1:
        y = y.replace('\\', '\\\\').replace('"', '\\"')
    
    return y, dict_distance_true, dict_distance_false


# Function to keep track of the true and false distance archives and update them
def keep_track(true_archive, false_archive, dict_distance_true, dict_distance_false, args_true_archive, args_false_archive, args, y):
    # Update the true archive with the new distances and arguments
    for idx in dict_distance_true:
        if idx not in true_archive:
            true_archive[idx] = dict_distance_true[idx]
            args_true_archive[idx] = (args, y)
        if dict_distance_true[idx] == 0:
            true_archive[idx] = dict_distance_true[idx]
            args_true_archive[idx] = (args, y)
    
    # Update the false archive with the new distances and arguments
    for idx in dict_distance_false:
        if idx not in false_archive:
            false_archive[idx] = dict_distance_false[idx]
            args_false_archive[idx] = (args, y)
        if dict_distance_false[idx] == 0:
            false_archive[idx] = dict_distance_false[idx]
            args_false_archive[idx] = (args, y)
    
    return true_archive, false_archive, args_true_archive, args_false_archive


# Function to update the pool with mutated or crossover individuals
def update_pool(pool, args):
    # Randomly decide whether to add a mutation or crossover to the pool
    if np.random.uniform() < 1/3:
        pool.append(mutation(args))  # Add mutated individual
    elif np.random.uniform() < 1/3:
        x, y = crossover(args, random.choice(pool))  # Add crossover individuals
        pool.append(x)
        pool.append(y)

# Function to generate test cases based on a dictionary of functions, expected values, and output folder
def write_test(temp_dict, val="true", folder="Fuzzy"):
    
    # Loop through each key in the dictionary (keys are filenames)
    for key in temp_dict:
        code = ""
        
        # Extract the original file name by splitting the key string
        original_file_name = key.split("/")[1].split("_instrumented")[0]
        print(f"Original file name: {original_file_name}")
        
        tuples = temp_dict[key]  # Get the function names and their corresponding data
        
        # If 'val' is "true", generate the test case imports and class definition
        if val == "true":
            for elem in tuples:
                f_name = elem[0].split("_instrumented")[0]  # Get function name
                code += f"from benchmark.{original_file_name} import {f_name}\n"
            
            # Add imports for unittest and define a test class
            code += "from unittest import TestCase\n"
            code += f"\nclass Test_{original_file_name}(TestCase):\n"
        
        # Loop through function names and their archives (branch information)
        for f_name, arch in tuples:
            # Loop through branches (test cases) in the archive
            for branch_nr in arch.keys():
                x, y = arch[branch_nr]  # Get input and expected output
                
                # Create a test function name dynamically
                test_f_name = f"test_{f_name}_{val}_{branch_nr}(self):\n"
                code += f"\tdef {test_f_name}"
                
                # Generate the function call with the arguments (x)
                code += f"\t\ty = {f_name.split('_instrumented')[0]}"

                args = x  # Arguments to pass to the function

                # Handle argument formatting based on the folder type (Fuzzy or other)
                if folder == "Fuzzy":
                    if isinstance(args, int):
                        code += f"({x})\n"
                    elif isinstance(args, str):
                        code += f'("{x}")\n'
                    else:
                        code += f"{x}\n"
                else:
                    # For other folder types, format multiple arguments as a tuple
                    args_s = "("
                    for arg in x:
                        if isinstance(arg, str):
                            arg = f"'{str(arg)}'"
                        args_s += f"{str(arg)},"
                    args_s = args_s.rstrip(",")  # Remove last comma
                    code += args_s + ")\n"
                
                # Add the assertion statement based on the expected output (y)
                if isinstance(y, str):
                    code += f"\t\tassert y == \"{y}\"\n\n"
                else:
                    code += f"\t\tassert y == {y}\n\n"
        
        # Determine file name and write the generated code to the specified folder
        file_name = f"{folder}/{original_file_name}_test.py"
        if val == "true":
            with open(file_name, 'w') as file:
                file.write(code)  # Overwrite the file for 'true' cases
        else:
            with open(file_name, 'a') as file:
                file.write("\n" + code)  # Append the code for 'false' cases


# Function for fuzzy testing, iterating over functions and their arguments
def fuzzy_testing(def_dict, dict_codes):
    avg_pool = []  # To store the average pool sizes across tests
    args_true_list, args_false_list = [], []  # To store true and false argument results
    
    # Loop through each file (key) in the dictionary
    for file in def_dict:
        file_path = file
        function_names = def_dict[file][::2]  # Get function names
        variable_types = def_dict[file][1::2]  # Get variable types for each function
        
        args_true = []  # Store true cases for the file
        args_false = []  # Store false cases for the file
        
        # Iterate through function names and their variable types
        for idx, f in enumerate(function_names):
            true_archive, false_archive = None, None  # Archives for true and false distances
            args_true_archive = {}  # Archive for tracking best true arguments
            args_false_archive = {}  # Archive for tracking best false arguments
            pool = create_pool(variable_types[idx])  # Create a pool of test cases
            
            # Loop through different sets of arguments for the function
            for args in pool:
                try:
                    # Execute the function and get the distances for true/false cases
                    y, dict_distance_true, dict_distance_false = execute_function(
                        dict_codes[file_path], f, args)
                    
                    # Initialize the archives if they're empty
                    if true_archive is None or false_archive is None:
                        true_archive = dict_distance_true.copy()
                        false_archive = dict_distance_false.copy()
                        
                        # Track the best arguments for true/false cases
                        args_true_archive = true_archive.copy()
                        for p in args_true_archive:
                            args_true_archive[p] = (args, y)
                        
                        args_false_archive = false_archive.copy()
                        for q in args_false_archive:
                            args_false_archive[q] = (args, y)
                    
                    else:
                        # Update the archives with new true/false cases
                        true_archive, false_archive, args_true_archive, args_false_archive = keep_track(
                            true_archive, false_archive, dict_distance_true, dict_distance_false, args_true_archive, args_false_archive, args, y)
                    
                    # Clear the distance dictionaries for the next iteration
                    dict_distance_true.clear()
                    dict_distance_false.clear()
                    
                    # Update the pool with mutated or crossover individuals
                    update_pool(pool, args)
                    avg_pool.append(len(pool))  # Track the pool size
                
                except AssertionError as e:
                    # If an assertion error occurs, continue to the next set of arguments
                    pass
                except BaseException as e:
                    # If a general error occurs, raise the exception
                    raise e
            
            # Store results for each function (true and false cases)
            args_true.append((f, dict(sorted(args_true_archive.items()))))
            args_false.append((f, dict(sorted(args_false_archive.items()))))
        
        # Store results for each file
        args_true_list.append((file_path, args_true))
        args_false_list.append((file_path, args_false))
    
    # Prepare data for writing the test cases
    temp_dict_true = {}
    for elem in args_true_list:
        key = elem[0]
        if key not in temp_dict_true:
            temp_dict_true[key] = []
        temp_dict_true[key].append(elem[1])
    
    # Flatten the dictionary for true cases
    for key in temp_dict_true:
        temp_dict_true[key] = temp_dict_true[key][0]
    
    temp_dict_false = {}
    for elem in args_false_list:
        key = elem[0]
        if key not in temp_dict_false:
            temp_dict_false[key] = []
        temp_dict_false[key].append(elem[1])
    
    # Flatten the dictionary for false cases
    for key in temp_dict_false:
        temp_dict_false[key] = temp_dict_false[key][0]
    
    # Write test cases for true and false conditions
    write_test(temp_dict_true, val="true", folder="Fuzzy")
    write_test(temp_dict_false, val="false", folder="Fuzzy")

# ------------------------ FUZZY ------------------------ #

# Get a list of file paths from the "parsed" folder
list_of_files = parsed_paths()

# Create a dictionary to store parsed code for each file, using the index as a key
my_dict = {idx: [] for idx in range(len(list_of_files))}

# Extract the code and the parsed code from the files
list_of_codes, dict_codes = parsed_codes(list_of_files)

# Populate the dictionary with the parsed code
for idx, code in enumerate(list_of_codes):
    my_dict[idx] = code

# Extract function signatures from the parsed code
my_dict = parsed_signatures(my_dict)

# Split each function signature into individual function definitions
for idx, elem in my_dict.items():
    if "new_signature" in elem:
        # Split the signature by "new_signature" and clean extra spaces
        splitted = elem.split("new_signature")
        splitted = [x.strip() for x in splitted]
        my_dict[idx] = splitted
    else:
        my_dict[idx] = [elem]

# Extract parameter types from function signatures
my_dict = get_types_of_parameters(my_dict)

# Organize extracted information into a tuple of function name and parameters
for idx, array in my_dict.items():
    new_tuple = []
    for t in array:
        function_name, param_types = t[0], t[1]
        function_name = function_name.split("(")[0].split(" ")[1]  # Clean function name
        param_types = [typ.strip() for typ in param_types]  # Clean parameter types
        new_tuple.append((function_name, param_types))
    my_dict[idx] = new_tuple

# Create a dictionary to store function definitions for each file
def_dict = {file: [] for file in list_of_files}

# Populate the function dictionary with the processed information
for idx, file in enumerate(def_dict):
    def_dict[file] = my_dict[idx]

# Testing function (commented out in the original code)
# testing_function(def_dict, dict_codes)

# Copy the "Fuzzy" folder into the "Archive" folder multiple times for testing
for i in range(1, COPIES + 1):
    print(f"Copy {i}")
    if not os.path.exists(f'Archive/fuzzer_test_archive/tests_fuzzer_copy_{i}'):
        fuzzy_testing(def_dict, dict_codes)
        original_folder = 'Fuzzy'
        destination_folder = f'Archive/fuzzer_test_archive/tests_fuzzer_copy_{i}'
        shutil.copytree(original_folder, destination_folder)

# ------------------------ DEAP ------------------------ #

# Variables for storing branch information and archive details
my_branches = None
archive_true_branches = {}  # To store true branch information
archive_false_branches = {}  # To store false branch information

# DEAP Hyperparameters (for Genetic Algorithm)
NPOP = 300  # Population size
NGEN = 10   # Number of generations
INDMUPROB = 0.05  # Individual mutation probability
MUPROB = 0.3  # Mutation probability
CXPROB = 0.3  # Crossover probability
TOURNSIZE = 3  # Tournament size
LOW = -1000  # Lower bound for variable values
UP = 1000  # Upper bound for variable values
REPS = 1  # Number of repetitions
MAX_STRING_LENGTH = 10  # Maximum length for strings

# DEAP creators for fitness and individual
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Function to create an individual for the genetic algorithm (GA)
def create_individual():
    global current_arg
    individual = []
    for elem in current_arg:
        if elem == "int":
            individual.append(init_int_variable())  # Create a random integer
        elif elem == "str":
            individual.append(init_str_variable())  # Create a random string
    return tuple(individual)  # Return as a tuple

# Normalize distance values for fitness calculation
def normalize(x):
    return x / (1.0 + x)

# Function to calculate fitness for an individual
def get_fitness(individual):
    global distances_true, distances_false
    global branches, archive_true_branches, archive_false_branches
    global current_f
    global my_branches

    # Reset distance dictionaries for true and false branches
    distances_true, distances_false = inst.get_distance_dicts()
    distances_true.clear()
    distances_false.clear()

    try:
        # Run the function with the provided arguments (x)
        x = individual[0]  # Get the argument
        if len(x) == 1:
            y = globals()[current_f](x[0])  # Single argument function call
        else:
            y = globals()[current_f](*x)  # Multiple arguments function call

        # If result is a string, escape special characters
        if isinstance(y, str) and len(y) >= 1:
            y.replace('\\', '\\\\').replace('"', '\\"')

    except AssertionError:
        # If function execution fails, return a high fitness value (bad solution)
        return float("inf"),
    except TypeError as e:
        raise e  # Raise the exception if a TypeError occurs

    # Get the number of branches for the current function
    my_branches = branches[current_f.split("_instrumented")[0]]

    # Initialize fitness value
    fitness = 0.0

    # Sum up normalized branch distances for true and false branches
    for branch in range(1, my_branches + 1):
        # For true branches
        if branch in distances_true:
            if distances_true[branch] == 0 and branch not in archive_true_branches:
                archive_true_branches[branch] = (x, y)  # Store branch information
            if branch not in archive_true_branches:
                fitness += normalize(distances_true[branch])

        # For false branches
        if branch in distances_false:
            if distances_false[branch] == 0 and branch not in archive_false_branches:
                archive_false_branches[branch] = (x, y)  # Store branch information
            if branch not in archive_false_branches:
                fitness += normalize(distances_false[branch])

    return fitness,  # Return fitness as a tuple

# Mutation function to randomly modify an individual
def mut(elem):
    individual = elem[0]
    if len(individual) == 1:
        individual = (mutation(individual[0]),)  # Mutate single-element individual
    else:
        individual = mutation(individual)  # Mutate multi-element individual
    elem[0] = individual
    return elem,

# Crossover function to combine two individuals into a child
def cross(elem1, elem2):
    parent1, parent2 = elem1[0], elem2[0]
    child1, child2 = crossover(parent1, parent2)  # Perform crossover
    elem1[0] = child1
    elem2[0] = child2
    return elem1, elem2

# Function to execute a code snippet within the DEAP environment
def execute_function_deap(code):
    current_module = sys.modules[__name__]
    code = compile(code, filename="<string>", mode="exec")
    exec(code, current_module.__dict__)  # Execute the compiled code

## Main function for the Genetic Algorithm (GA)
def GA_deap():
    # Access global variables that store information about branches and coverage
    global archive_true_branches, archive_false_branches
    global list_true_archive, list_false_archive
    global current_f, file_path
    
    # Create a DEAP toolbox to hold the evolutionary operators
    toolbox = base.Toolbox()
    
    # Register the individual creation function (each individual is generated by 'create_individual')
    toolbox.register("attr_str", create_individual)
    
    # Register the initialization method for an individual (using 'create_individual' to generate attributes)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_str, n=1)
    
    # Set the population (a list of individuals)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define the fitness evaluation function (used to calculate fitness of an individual)
    toolbox.register("evaluate", get_fitness)
    
    # Register the crossover (mating) operator
    toolbox.register("mate", cross)
    
    # Register the mutation operator
    toolbox.register("mutate", mut)
    
    # Register the selection operator (tournament selection with defined tournament size)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    
    # Initialize an empty list to track the coverage during each repetition
    coverage = []
    
    # Run the genetic algorithm for a specified number of repetitions (REPS)
    for i in range(REPS):
        # Reset archives of true and false branches for each repetition
        archive_true_branches = {}
        archive_false_branches = {}
        
        # Create the initial population of individuals (NPOP size)
        population = toolbox.population(n=NPOP)
        
        # Run the evolutionary algorithm (simple EA) with the toolbox configuration
        algorithms.eaSimple(population, toolbox, CXPROB, MUPROB, NGEN, verbose=False)
        
        # Calculate coverage (the sum of true and false branches)
        cov = len(archive_true_branches) + len(archive_false_branches)
        
        # Store the coverage value for later analysis
        coverage.append(cov)
    
    # After the genetic algorithm run, store the archive of true and false branches for the current file and function
    list_true_archive.append((file_path, current_f, dict(sorted(archive_true_branches.items()))))
    list_false_archive.append((file_path, current_f, dict(sorted(archive_false_branches.items()))))

# ------------------------ DEAP TESTING AND ARCHIVING ------------------------ #

# For each file, run the GA and save the test in "*file_name*_test.py", then copy the 'Deap' folder to the archive
for i in range(1, COPIES + 1):
    print(f"Copy {i}")
    list_true_archive, list_false_archive = [], []  # Clear previous archives
    
    # Loop through each instrumented file
    for instrumented_file in dict_codes:
        file_path = instrumented_file  # Get the file path
        code = dict_codes[instrumented_file]  # Get the code of the file
        execute_function_deap(code)  # Execute the function in the DEAP environment
        
        # Extract function names and their corresponding parameter types
        function_names = def_dict[instrumented_file][::2]
        parameters_type = def_dict[instrumented_file][1::2]
        
        # Create temporary lists to store archives and coverage data for the current file
        true_archive, false_archive, coverage_archive = [], [], []
        
        # For each function in the file, run the GA and collect data on covered branches
        for current_f in function_names:
            current_arg = parameters_type[function_names.index(current_f)]  # Get argument types for the function
            branches = inst.get_branches_dict()  # Get branch information for the function
            my_branches = branches[current_f.split("_instrumented")[0]]  # Get branches for the un-instrumented function
            GA_deap()  # Run the Genetic Algorithm (GA) on this function
    
    # Create temporary dictionaries to group true and false branches by file
    temp_dict_true = {}
    for elem in list_true_archive:
        key = elem[0]
        if key not in temp_dict_true:
            temp_dict_true[key] = []
        temp_dict_true[key].append((elem[1], elem[2]))  # Store function and true branches in the dictionary
    
    temp_dict_false = {}
    for elem in list_false_archive:
        key = elem[0]
        if key not in temp_dict_false:
            temp_dict_false[key] = []
        temp_dict_false[key].append((elem[1], elem[2]))  # Store function and false branches in the dictionary
    
    # Write the test scripts for true and false branches into "*file_name*_test.py"
    write_test(temp_dict_true, val="true", folder="Deap")
    write_test(temp_dict_false, val="false", folder="Deap")
    
    # Copy the 'Deap' folder into the 'Archive' directory for storage
    if not os.path.exists(f'Archive/fuzzer_test_archive/tests_fuzzer_copy_{i}'):
        original_folder = 'Deap'
        destination_folder = f'Archive/deap_test_archive/tests_deap_copy_{i}'
        shutil.copytree(original_folder, destination_folder)

# ------------------------ STATISTICAL COMPARISON ------------------------ #

# Function to load the JSON data from a file (mutation scores)
def get_json_data(file_path):
    """
    Parameters:
    - file_path: Path to the file containing JSON data.
    
    Returns:
    - Parsed JSON data.
    """
    # Open the file and read its contents as JSON
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

# Function to compute Cohen's d effect size for comparing two groups
def cohen_d(group1, group2):
    """
    Parameters:
    - group1: First group of scores (e.g., Fuzzer scores).
    - group2: Second group of scores (e.g., DEAP scores).
    
    Returns:
    - Cohen's d effect size: A measure of the difference between the two groups.
    """
    mean_diff = np.mean(group1) - np.mean(group2)  # Difference in means
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)  # Pooled standard deviation
    effect_size = mean_diff / pooled_std  # Cohen's d formula
    return effect_size

# Function to plot boxplots comparing Fuzzer and DEAP scores
def plot_boxplots(df, name, fuzzer_avg, deap_avg, cohen_d_value, p_value, wilcoxon_result):
    """
    Parameters:
    - df: Dataframe with Fuzzer and DEAP scores for plotting.
    - name: Name of the file for the plot.
    - fuzzer_avg: Average Fuzzer score.
    - deap_avg: Average DEAP score.
    - cohen_d_value: Cohen's d effect size value.
    - p_value: p-value of the Wilcoxon test.
    - wilcoxon_result: Result of the Wilcoxon test.
    
    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a new plot
    
    # Create the boxplot for Fuzzer and DEAP scores
    sns.boxplot(ax=ax, x='File_Name', y='Score', hue='Metric', data=df)
    
    # Set axis labels and plot title
    ax.set_xlabel('File Name')
    ax.set_ylabel('Mutation Score')
    ax.set_title(f'Mutation Score of {name}\nFuzzer Avg: {round(fuzzer_avg, 3)}\nDEAP Avg: {round(deap_avg, 3)}\nCohen\'s d: {round(cohen_d_value, 3)}\np-value: {p_value}: {wilcoxon_result}')
    
    plt.tight_layout()  # Adjust layout for better spacing
    fig.savefig(f"Boxplots/{name}.png")  # Save the plot as a PNG file

# Function to perform statistical comparison between Fuzzer and DEAP scores
def compute_statistical_comparison(json_data):
    """
    Parameters:
    - json_data: JSON data containing the scores for Fuzzer and DEAP.
    
    Returns:
    - None
    """
    for file, scores in json_data.items():
        name = file.split('.')[0]  # Get the base name of the file
        fuzzer_scores = scores[0]  # Fuzzer scores
        deap_scores = scores[1]  # DEAP scores
        
        # Create a combined dataframe for plotting
        combined_data = [(name, score, 'Fuzzer') for score in fuzzer_scores] + \
                        [(name, score, 'Deap') for score in deap_scores]
        df = pd.DataFrame(combined_data, columns=['File_Name', 'Score', 'Metric'])
        
        # Compute Cohen's d (effect size) between Fuzzer and DEAP scores
        cohen_d_value = cohen_d(fuzzer_scores, deap_scores)
        
        # Perform the Wilcoxon test to check if there is a significant difference between the two sets of scores
        statistic, p_value = wilcoxon(fuzzer_scores, deap_scores, zero_method='zsplit')
        
        # Interpret the result of the Wilcoxon test
        alpha = 0.05
        wilcoxon_result = 'Significant difference' if p_value < alpha else 'No significant difference'
        
        # Plot the boxplot for visual comparison
        plot_boxplots(df, name, np.mean(fuzzer_scores), np.mean(deap_scores), cohen_d_value, round(p_value, 3), wilcoxon_result)

# Load JSON data from mutation_scores.txt and perform statistical comparison
file_path = 'mutation_scores.txt'
json_data = get_json_data(file_path)
compute_statistical_comparison(json_data)

