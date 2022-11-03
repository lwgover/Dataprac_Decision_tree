# Dataprac_Decision_tree
Through data from American responses to the World Values Survey, I construct a decision tree for a given dependent variable.

This program constructs a Decision Tree / Regression Tree from a spreadsheet and code book. The included Spreadsheet and Codebook come from the Dataprac: data from the World Values Survey limited to Americans and only certain political science questions to construct a regression tree.
This tree is formed through a modified version of Iterative Dichotomiser 3 (ID3). Due to the program’s variable structure, it constructs a Regression tree for Ordinal and Ratio type questions, and a Decision tree for Nominal type questions.


Algorithm: Modified ID3
A = List of Individuals, each with a DV response a and IV responses {1..n}
Calculate total entropy / variance across all individuals
Find the question that, if split by it, most reduces entropy / variance
split individuals into 2 or more groups based on their response to that question recursively do this for each sub-group, until there is no information gained / variance lost


Usage: The command line takes 3 required arguments, data file, codebook file, and a Dependent Variable’s dataprac number
Sample: python3 decision_tree.py DatapracRStudio.xlsx DatapracCodebook.xlsx DP54

Optional: You may follow this with a list of independent variables that the tree will consider when it makes the tree
Sample: python3 decision_tree.py DatapracRStudio.xlsx DatapracCodebook.xlsx DP54 DP1 DP2 DP3 DP5
