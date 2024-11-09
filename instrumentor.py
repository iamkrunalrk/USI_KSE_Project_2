import ast  # Used for parsing and manipulating Python code's abstract syntax tree (AST)
import os   # Provides functionality for interacting with the operating system, e.g., file handling

from nltk.metrics.distance import edit_distance  # Importing edit distance function from nltk

# ------------------------ Dictionaries ------------------------ #

# These dictionaries store the distances for 'true' and 'false' conditions encountered during analysis
distance_dict_true = {}  # Stores distances when the condition evaluates to true
distance_dict_false = {}  # Stores distances when the condition evaluates to false

# This dictionary keeps track of the number of branches for each function
branches_dict = {}

# Function to return both distance dictionaries
def get_distance_dicts():
    return distance_dict_true, distance_dict_false

# ---------------------- Original filename ------------------- #

# Function that returns the list of files found in the benchmark directory
def get_original_filenames():
    return list_of_files

# Function that returns the dictionary containing branch numbers for each function
def get_branches_dict():
    return branches_dict

# ------------------------ Transformer ------------------------ #

# Custom AST NodeTransformer class to modify Python code's AST
class Transformer(ast.NodeTransformer):
    # Initializing class variables
    branch_num = 0
    name_lists = []  # Stores function names
    node_name = ""
    number_of_functions = 0  # Counter for functions found
    number_of_comparisons = 0  # Counter for comparisons found

    # Function to handle function definitions (FunctionDef nodes)
    def visit_FunctionDef(self, node):
        # Reset branch number for each new function
        self.branch_num = 0
        
        # Add the function name to the list
        self.name_lists.append(node.name)
        
        # Update the current node name
        self.node_name = node.name
        
        # Assign the current function a branch number (initially 0)
        branches_dict[self.node_name] = self.branch_num
        
        # Modify the function name by adding "_instrumented"
        node.name = node.name + "_instrumented"
        
        # Increment the function counter
        self.number_of_functions += 1
        
        # Continue visiting child nodes
        return self.generic_visit(node)
    
    # Function to handle comparison nodes (Compare nodes)
    def visit_Compare(self, node):
        # Skip comparison if the operator is a membership or identity check
        if node.ops[0] in [ast.Is, ast.IsNot, ast.In, ast.NotIn]:
            return node
        
        # Increment the branch number for each comparison
        self.branch_num += 1
        branches_dict[self.node_name] = self.branch_num
        
        # Increment the comparison counter
        self.number_of_comparisons += 1
        
        # Replace the comparison with a call to the evaluate_condition function
        return ast.Call(
            func=ast.Name("evaluate_condition", ast.Load()),
            args=[ast.Num(self.branch_num), 
                  ast.Str(node.ops[0].__class__.__name__),  # Operator type
                  node.left,  # Left operand of the comparison
                  node.comparators[0]],  # Right operand of the comparison
            keywords=[],
            starargs=None,
            kwargs=None
        )
    
    # Function to handle assert nodes (Assert nodes)
    def visit_Assert(self, node):
        return node  # No change to assert nodes
    
    # Function to handle call nodes (Call nodes)
    def visit_Call(self, node):
        try:
            # If the called function is in the name list, modify it by adding "_instrumented"
            if node.func.id in self.name_lists:
                node.func.id = node.func.id + "_instrumented"
        except AttributeError:
            pass  # Ignore if the node is not a function call with an id
        
        return node
    
    # Function to handle return nodes (Return nodes)
    def visit_Return(self, node):
        # If the return value is a function call, continue traversing the AST
        if isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        else:
            return node  # No modification for non-call return values

# ------------------------ evaluate_condition ------------------------ #

# Function to evaluate conditions and update distance dictionaries
def evaluate_condition(num, op, lhs, rhs):
    # Initialize distances for true and false conditions
    distance_true = 0
    distance_false = 0
    
    # Track lengths for string comparisons
    long_l = -1
    long_r = -1
    
    # Convert single-character strings to their ASCII values
    if isinstance(lhs, str):
        if len(lhs) == 1:
            lhs = ord(lhs)  # Convert to ASCII value
        else:
            long_l = len(lhs)  # Store string length
    
    if isinstance(rhs, str):
        if len(rhs) == 1:
            rhs = ord(rhs)  # Convert to ASCII value
        else:
            long_r = len(rhs)  # Store string length
    
    # Handle string comparisons
    if long_l != -1 or long_r != -1:
        if op == "Eq":
            distance_true = edit_distance(lhs, rhs)
            distance_false = 1 if lhs == rhs else 0
        elif op == "NotEq":
            distance_true = 1 if lhs == rhs else 0
            distance_false = edit_distance(lhs, rhs)
    else:
        # Handle numeric comparisons
        if op == "Lt":
            distance_true = lhs - rhs + 1 if lhs >= rhs else 0
            distance_false = rhs - lhs if lhs < rhs else 0
        elif op == "Gt":
            distance_true = rhs - lhs + 1 if lhs <= rhs else 0
            distance_false = lhs - rhs if lhs > rhs else 0
        elif op == "LtE":
            distance_true = lhs - rhs if lhs > rhs else 0
            distance_false = rhs - lhs + 1 if lhs <= rhs else 0
        elif op == "GtE":
            distance_true = rhs - lhs if lhs < rhs else 0
            distance_false = lhs - rhs + 1 if lhs >= rhs else 0
        elif op == "Eq":
            distance_true = abs(lhs - rhs) if lhs != rhs else 0
            distance_false = 1 if lhs == rhs else 0
        elif op == "NotEq":
            distance_true = 1 if lhs == rhs else 0
            distance_false = abs(lhs - rhs) if lhs != rhs else 0
    
    # Update the distance dictionaries with the calculated distances
    update_maps(num, distance_true, distance_false)
    
    # Return True if the true distance is 0, else return False
    return True if distance_true == 0 else False

# ------------------------ update_maps ------------------------ #

# Function to update the distance dictionaries for a given condition
def update_maps(condition_num, d_true, d_false):
    # Update the dictionary storing true distances
    if condition_num in distance_dict_true:
        # Keep the larger of the current value and the new one
        if distance_dict_true[condition_num] <= d_true:
            distance_dict_true[condition_num] = distance_dict_true[condition_num]
    else:
        distance_dict_true[condition_num] = d_true
    
    # Update the dictionary storing false distances
    if condition_num in distance_dict_false:
        # Keep the larger of the current value and the new one
        if distance_dict_false[condition_num] <= d_false:
            distance_dict_false[condition_num] = distance_dict_false[condition_num]
    else:
        distance_dict_false[condition_num] = d_false

# ------------------------ Main ------------------------ #

# Path to the benchmark directory
files_in_benchmark = "benchmark"
list_of_files = []

# Loop through all files in the benchmark directory
for filename in os.listdir(files_in_benchmark):
    file_path = os.path.join(files_in_benchmark, filename)
    
    # Only process files, not directories
    if os.path.isfile(file_path):
        list_of_files.append(file_path)

# Create an instance of the Transformer class
transformer = Transformer()

# Process each file in the benchmark directory
for file_path in list_of_files:
    with open(file_path, "r") as code_file:
        code_content = code_file.read()
    
    # Parse the code into an AST and apply the transformer
    tree = ast.parse(code_content)
    transformed_tree = transformer.visit(tree)
    
    # Unparse the transformed AST back into Python code
    transformed_code = ast.unparse(transformed_tree)
    
    # Extract the file name without the directory or extension
    filename_without_ext = os.path.basename(file_path).replace(".py", "")
    
    # Create the path for the new instrumented file
    instrumented_file_path = f"parsed/{filename_without_ext}_instrumented.py"
    
    # Write the instrumented code to the new file
    with open(instrumented_file_path, 'w') as output_file:
        output_file.write("from instrumentor import evaluate_condition\n\n\n")
        output_file.write(transformed_code)

# Print summary of processing results
print("Number of files in the parsed folder:", len(list_of_files))
print("Number of functions found:", transformer.number_of_functions)
print("Number of comparisons found:", transformer.number_of_comparisons)
