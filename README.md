# Dataprac Decision Tree
Through data from American responses to the World Values Survey, I construct a decision tree for a given dependent variable.

This program constructs a Decision Tree / Regression Tree from a spreadsheet and code book. The included Spreadsheet and Codebook come from the Dataprac: data from the World Values Survey limited to Americans and only certain political science questions to construct a regression tree.
This tree is formed through a modified version of Iterative Dichotomiser 3 (ID3). Due to the variety of data types asked for in questions, the program constructs a Regression tree for Ordinal and Ratio type questions, and a Decision tree for Nominal type questions.


## Algorithm: 
```
Modified ID3

A = List of Individuals, each with a DV response a and IV responses {1..n}
Calculate total entropy / variance across all individuals
Find the question that, if split by it, most reduces entropy / variance
split individuals into 2 or more groups based on their response to that question recursively do this for each sub-group, until there is no information gained / variance lost
```

## Usage: 
The command line takes 3 required arguments, data file, codebook file, and a Dependent Variable’s dataprac number
Sample: `python3 decision_tree.py DatapracRStudio.xlsx DatapracCodebook.xlsx DP54`

Optional: You may follow this with a list of independent variables that the tree will consider when it makes the tree
Sample: `python3 decision_tree.py DatapracRStudio.xlsx DatapracCodebook.xlsx DP54 DP1 DP2 DP3 DP5`

Sample Interaction:

```
python3 decision_tree.py DatapracRStudio.xlsx DatapracCodebook.xlsx DP54 DP1 DP5    
Biological sex: 
		1 male: Education: 
				1 less than high school: 5.8: closer to there should be greater incentives for individual effort
				4 2-year degree: 6.021739130434782: closer to there should be greater incentives for individual effort
				5 4-year degree: 6.616071428571429: closer to there should be greater incentives for individual effort
				3 some college: 6.6231884057971016: closer to there should be greater incentives for individual effort
				8 Professional (JD/MD): 6.7272727272727275: closer to there should be greater incentives for individual effort
				2 high school/GED: 6.7625: closer to there should be greater incentives for individual effort
				6 Masters: 6.927536231884058: closer to there should be greater incentives for individual effort
				7 Doctorate: 7: closer to there should be greater incentives for individual effort

		2 female: Education: 
				1 less than high school: 3.7142857142857144: closer to incomes should be more equal 
				2 high school/GED: 5.290909090909091: closer to incomes should be more equal 
				3 some college: 5.6138613861386135: closer to there should be greater incentives for individual effort
				5 4-year degree: 5.872093023255814: closer to there should be greater incentives for individual effort
				6 Masters: 6: closer to there should be greater incentives for individual effort
				4 2-year degree: 6.205128205128205: closer to there should be greater incentives for individual effort
				8 Professional (JD/MD): 6.636363636363637: closer to there should be greater incentives for individual effort
				7 Doctorate: 7.090909090909091: closer to there should be greater incentives for individual effort
```

This implementation is currently too slow to run on large datasets like the full World Values Survey, but I plan to optimize it and rewrite it as a multi-threaded `c` program at some point in the future.
