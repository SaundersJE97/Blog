Decision Tree
========



# Using [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) (principle Component Analysis) to plot information

# Calculating Entropy

1. Calculate the unique elements of an array
2. Find the probability that a class exists within the sample (n/N)
3. Calculate entropy
$-\sum_{i=0}^n(p_i * log_2(p_i))$

# Finding the split

Use the function calculate_entropy to complete the find_split function

We find the optimal feature and the best value to tpslit the data into two chunks.  We then repeat this on both of the new
data

# Building the tree

We use recursion to build a decision tree.  But it needs a stopping condition.  We will use two.

1. Maximum depth: tree is limited to be no deeper than a provided limit
2. Perfection: if a node contans only one class

Hence using recursion, we can use build_tree(x, y, max_depth) to build the tree.

To prevent dividing the data forever, we use

# Prediction

Traverse the tree using recursion

# Resources:
Awesome Decision Tree Research Papers:
https://github.com/benedekrozemberczki/awesome-decision-tree-papers
A curated list of classification and regression tree research papers with implementations from top conferences.