"""
Makes a decision tree for dataprac data
"""
__author__ = "Lucas Gover" 
__license__ = "University of Puget Sound"
__date__ = "April 2022"

import sys
import math
import numpy as np
import pandas as pd
import statistics
import random
#==========================================================================================================
#                                              Classes
#==========================================================================================================

"""
    Class containing all the responses of one person
"""
class Individual:
    def __init__(self, responses):
        self.responses = responses
    def get_responses(self): return self.responses
    def get_response(self,index): return self.responses[index]

"""
    Class containing all the information of every variable in the analysis
"""
class Codebook:
    # list of number_meanings that give you (WVS_number, Dataprac_number, Question_Label, min, max, variable_type(Nominal, binary, Ordinal, Ratio), numbers and their meanings]
    def __init__(self,file):
        self.descriptions = self._parse(file)
        
    def _parse(self, file:str) -> list:
        descriptions_list = list(pd.ExcelFile(file).parse().to_numpy())
        descriptions =list(map(self._parse_line,descriptions_list))
        return descriptions

    def _parse_line(self,line) -> tuple:
        #list of number_meanings that give you (WVS_number, Dataprac_number, Question_Label, min, max, variable_type(Nominal, binary, Ordinal, Ratio), if nominal or binary, meaning behind those numbers, if ordinal or ratio, meaining behind high and low)
        WVS_Number = str(line[0])
        dataprac_number = str(line[1])
        question_label = str(line[2])
        variable_type = str(line[4])
        numbers_and_meanings = self._isolate_relevant_numbers_with_meanings(line[3],variable_type)
        min = numbers_and_meanings[0][0]
        max = numbers_and_meanings[-1][0]
        output = dict()
        output["WVS_Number"] = WVS_Number
        output["dataprac_number"] = dataprac_number
        output["question_label"] = question_label
        output["variable_type"] = variable_type
        output["numbers_and_meanings"] = numbers_and_meanings
        output["min"] = min
        output["max"] = max
        return output

    def _isolate_relevant_numbers_with_meanings(self,coding:str,var_type:str) -> list:
        """
            Interprets the meanings of the variable descriptions from excel
        """
        numbers = []
        meanings = []
        if 'Ordinal'in var_type or 'Ratio'in var_type:
            if ',' in coding:
                numbers, meanings = self._split_numbers_meanings(coding,',')
            elif '-' in coding:
                numbers, meanings = self._split_numbers_meanings(coding,'-')
        if ('Nominal' in var_type) or ('binary' in var_type):
            numbers, meanings = self._split_numbers_meanings(coding,',',is_nominal=True)

        numbers_and_meanings = list(zip(numbers,meanings))
        return numbers_and_meanings

    def _split_numbers_meanings(self,coding, split, is_nominal = False):
        """
            Returns a list of tuples with 
            [0] as an int representing the number
            [1] as a string representation of the meaning of the number
        """
        numbers = []
        meanings = []
        codings = coding.split(split)
        looking_for_number = True
        looking_for_meaning = True
        for code in codings:
            index = 0
            current_word = ""
            if looking_for_number:
                #find first digit
                while not code[index].isdigit():
                    index += 1
                #get_all digits in number
                while index < len(code) and code[index].isdigit():
                    current_word += code[index]
                    index += 1
                #add Number
                numbers.append(int(current_word))
                current_word = ""
                looking_for_number = False
                looking_for_meaning = True
            if looking_for_meaning:
                while index < len(code) and (not code[index].isalpha()) and (not code[index].isdigit()):
                    index += 1
                    current_word += code[index]
                if is_nominal and (len(code) - index < 2):
                    continue
                #add meaning
                meanings.append(code[index:])
                looking_for_number = True
                looking_for_meaning= False
        return (numbers,meanings)

    
    def get_meaning_of_number(self, variable, number) -> str:
        """
        Because many of the ordinal and ratio values will end up being decimals, which aren't coded for in the codebook,
        this program gets the number closest to any input number and returns its meaning
        """
        numbers_and_meanings = self.descriptions[variable]["numbers_and_meanings"]

        closest_number = -1
        closest_number_difference = 1000000
        closest_number_meaning = ""
        for nm in numbers_and_meanings:
            difference = abs(nm[0] - number)
            if difference < closest_number_difference:
                closest_number_difference = difference
                closest_number = nm[0]
                closest_number_meaning = nm[1]

        if closest_number == number:
            return closest_number_meaning
        return "closer to " + closest_number_meaning

    def get_index_of_WVS_number(self,WVS_number:str) -> int:
        for i in range(len(self.descriptions)):
            if self.descriptions[i]["WVS_Number"] == WVS_number:
                return i
        raise Exception("Couldn't find " + WVS_number)

    def get_index_of_dataprac_number(self,dataprac_number:str) -> int:
        for i in range(len(self.descriptions)):
            if self.descriptions[i]["dataprac_number"] == dataprac_number:
                return i
        raise Exception("Couldn't find " + dataprac_number)

    def get_question(self,variable_index:int) -> str:
        return self.descriptions[variable_index]["question_label"]

    def get_min_of_variable(self,variable_index:int) -> int:
        return self.descriptions[variable_index]["min"]

    def get_max_of_variable(self,variable_index:int) -> int:
        return self.descriptions[variable_index]["max"]

    def get_datatype_of_variable(self,variable_index:int) -> int:
        return self.descriptions[variable_index]["variable_type"]

    def get_significant_numbers(self,variable_index):
        return list(map(lambda meanings:meanings[0],self.descriptions[variable_index]["numbers_and_meanings"]))

    def get_dataprac_number(self,num:int) -> int:
            return self.descriptions[num]["dataprac_number"]

class Decision_Tree:
    def __init__(self,individuals,parent_majority, codebook:Codebook = None,DV_index = 0,DV_type = None, variable_index = 0,variable_description=None, is_leaf=True, children = None):
        self.individuals = individuals #list of individuals that the decision tree is deciding on
        self.variable_description = variable_description # verbal description of the variable that the decision tree is splitting on
        self.DV_type = DV_type # nominal, ratio, ordinal, binary ... 
        self.DV_index = DV_index # index in the codebook and individuals that the DV is at
        self.is_leaf = is_leaf #whether is decision tree is a leaf or not
        self.variable_index = variable_index # index of the variable this decision tree splits on
        self.codebook = codebook # Codebook
        self.children = children # (classifier, verbal description of classifier function, significance of split, tree)
        self.classification = self.init_classification(parent_majority)

    def classify(self,individual:Individual):
        """
        Classifies a given individual
        
        """
        if self.is_leaf:
            return self.classification
        #if it's a nominal or binary variable
        if 'Nominal' in self.codebook.get_datatype_of_variable(self.variable_index) or 'binary' in self.codebook.get_datatype_of_variable(self.variable_index):
            for child in self.children:
                if individual.responses[self.variable_index] == child[0]:
                    return child[3].classify(individual)
            return self.classification # if the decision tree doesn't have a branch for this person, return the classification of this tree

        #if ordinal or ratio variable
        if 'Ordinal' in self.codebook.get_datatype_of_variable(self.variable_index) or 'Ratio' in self.codebook.get_datatype_of_variable(self.variable_index):
            if individual.responses[self.variable_index] < self.children[0][0]:
                return self.children[0][3].classify(individual)
            else:
                return self.children[1][3].classify(individual)
            
        raise Exception("Cannot Classify Item")

    def init_classification(self,parent_majority):
        """
        Initial Classification of an element. If there's a tie it will give it to the majority of the parent node
        """
        if len(self.individuals) == 0:
            return parent_majority
        if 'Nominal' in self.DV_type or 'binary' in self.DV_type:
            responses = map(lambda ind: ind.responses[self.DV_index],self.individuals)
            return statistics.mode(responses)
        if 'Ordinal' in self.DV_type or 'Ratio' in self.DV_type:
            responses = map(lambda ind: ind.responses[self.DV_index],self.individuals)
            return statistics.mean(responses)

    def print_tree(self, depth):
        """
        Prints tree 
        """
        if self.is_leaf:
            print(str(self.classification) + ": " + self.codebook.get_meaning_of_number(self.DV_index,self.classification))
        else:
            print(self.variable_description + ": ")
            self.children.sort(key=lambda x: x[3].classification)
            for branch in self.children:
                print("\t\t" * (depth + 1),end = '')
                print(str(branch[1]) +" " +  str(branch[2]) + ": ",end='')
                branch[3].print_tree(depth+1)
            print()
    """
    Limits the depth of the tree
    """
    def limit_depth(self,depth:int):
        if depth == 0:
            self.is_leaf = True
        if self.children == None:
            return
        else:
            for child in self.children:
                child[3].limit_depth(depth-1)

#===============================================================================================================================================================
def make_nominal_decision_tree(individuals:list, codebook:Codebook, DV_index, parent_majority) ->  Decision_Tree:
    """Creates a decision tree using an information gain approach"""
    #Check if there are any individuals left to check
    if len(individuals) == 0: return Decision_Tree(individuals,parent_majority,codebook = codebook, DV_index = DV_index)
    
    #if is uniformly labeled return self
    if len(list(filter(lambda ind:ind.responses[DV_index] == individuals[0].responses[DV_index],individuals))) == 0:
        return Decision_Tree(individuals,parent_majority,codebook = codebook, DV_index = DV_index)

    #calculate entropy of entire training set
    total_entropy = entropy(individuals,DV_index)

    #find split that most reduces entropy
    best_variable = -1
    best_splits = None
    best_information_gain = -99999999
    for i in range(len(individuals[0].responses)):
        if i == DV_index:
            continue
        splits = get_entropy_splits(individuals,DV_index,i,codebook.get_datatype_of_variable(i),codebook) #(classifier function, individuals)
        lists_of_individuals = list(map(lambda split: split[1],splits))
        information_gain = total_entropy - split_entropy(lists_of_individuals,len(individuals),DV_index)

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_splits = splits
            best_variable = i

    if best_information_gain <= 0:
        return Decision_Tree(individuals,parent_majority,DV_type = codebook.get_datatype_of_variable(DV_index),DV_index = DV_index,codebook = codebook)

    new_parents_majority = most_common_response(individuals,DV_index)

    #make children on best split
    branches = []
    first = True
    for i in best_splits:
        #classifier function, verbal description of classifier function, significance of split, tree)
        IV_type = codebook.get_datatype_of_variable(best_variable)
        if 'Nominal' in IV_type or 'binary' in IV_type:
            classification = i[0]
            classifier_description = str(i[0])
            meaning_of_split = codebook.get_meaning_of_number(best_variable,i[0])
            sub_tree = make_nominal_decision_tree(i[1],codebook,DV_index,new_parents_majority)
            branch = (classification,classifier_description,meaning_of_split,sub_tree)
        if 'Ordinal' in IV_type or 'Ratio' in IV_type:
            if first:
                classification = i[0]
                classifier_description = "<" + str(i[0])
                meaning_of_split = get_ordinal_meaining(codebook,best_variable,i[0],True)
                first = False
            else: 
                classification = i[0]
                classifier_description = ">= " + str(i[0])
                meaning_of_split = get_ordinal_meaining(codebook,best_variable,i[0],False)
            #if get_ordinal_meaining(codebook,best_variable,i[0],True) == get_ordinal_meaining(codebook,best_variable,i[0],False):
                #raise Exception("Get Ordinal Meaning not working lol")
            sub_tree = make_nominal_decision_tree(i[1],codebook,DV_index,new_parents_majority)
            branch = (classification,classifier_description,meaning_of_split,sub_tree)
        branches.append(branch)
    return Decision_Tree(individuals,parent_majority,DV_type='Nominal',DV_index = DV_index,codebook = codebook,variable_description = codebook.get_question(best_variable),is_leaf=False,children=branches,variable_index=best_variable)


def make_ordinal_decision_tree(individuals:list, codebook:Codebook, DV_index, parent_majority) ->  Decision_Tree:
    """Creates a decision tree using an variance minimizing approach"""
    #Check if there are any individuals left to check
    if len(individuals) == 0: return Decision_Tree(individuals,parent_majority,codebook = codebook, DV_index = DV_index)
    
    #if is uniformly labeled return self
    if len(list(filter(lambda ind:ind.responses[DV_index] == individuals[0].responses[DV_index],individuals))) == 0:
        return Decision_Tree(individuals,parent_majority,codebook = codebook, DV_index = DV_index)

    #calculate entropy of entire training set
    total_variance = variance(individuals,DV_index)

    #find split that most reduces entropy
    best_variable = -1
    best_splits = None
    best_variance = total_variance
    for i in range(len(individuals[0].responses)):
        if i == DV_index:
            continue
        splits = get_variance_splits(individuals,DV_index,i,codebook.get_datatype_of_variable(i),codebook)
        lists_of_individuals = list(map(lambda split: split[1],splits))
        variance_split = split_variance(lists_of_individuals,len(individuals),DV_index)
        #print("total variance: " + str(total_variance))
        #print("split_variance: " + str(variance_split))

        if variance_split < best_variance:
            best_variance = variance_split
            best_splits = splits
            best_variable = i

    if total_variance <= best_variance:
        return Decision_Tree(individuals,parent_majority,DV_type = codebook.get_datatype_of_variable(DV_index),DV_index = DV_index,codebook = codebook)

    new_parents_majority = most_common_response(individuals,DV_index)

    #make children on best split
    branches = []
    first = True
    for i in best_splits:
        #classifier function, verbal description of classifier function, significance of split, tree)
        IV_type = codebook.get_datatype_of_variable(best_variable)
        if 'Nominal' in IV_type or 'binary' in IV_type:
            classification = i[0]
            classifier_description = str(i[0])
            meaning_of_split = codebook.get_meaning_of_number(best_variable,i[0])
            sub_tree = make_nominal_decision_tree(i[1],codebook,DV_index,new_parents_majority)
            branch = (classification,classifier_description,meaning_of_split,sub_tree)
        if 'Ordinal' in IV_type or 'Ratio' in IV_type:
            if first:
                classification = i[0]
                classifier_description = "<" + str(i[0])
                meaning_of_split = get_ordinal_meaining(codebook,best_variable,i[0],True)
                first = False
            else: 
                classification = i[0]
                classifier_description = ">= " + str(i[0])
                meaning_of_split = get_ordinal_meaining(codebook,best_variable,i[0],False)
            sub_tree = make_nominal_decision_tree(i[1],codebook,DV_index,new_parents_majority)
            branch = (classification,classifier_description,meaning_of_split,sub_tree)
        branches.append(branch)
    return Decision_Tree(individuals,parent_majority,DV_type='Ordinal',DV_index = DV_index,codebook = codebook,variable_description = codebook.get_question(best_variable),is_leaf=False,children=branches,variable_index=best_variable)

def most_common_response(individuals, DV):
    """
    Returns the most common response a group of individuals said in respect to a given DV
    """
    responses = dict()

    for i in individuals:
        if not (i.get_response(DV) in responses.keys()):
            responses[i.get_response(DV)] = 0
        responses[i.get_response(DV)] += 1

    most_common = None
    biggest_num = -1
    for i in responses.keys():
        if responses[i] > biggest_num:
            biggest_num = responses[i]
            most_common = i
    return most_common
        
def variance(individuals, DV_index):
    """
    Calculates the variance contained in a list of individuals with respect to a given dependent variable
    """
    if len(individuals) <= 1:
        return 0

    #get mean
    mean = 0
    for ind in individuals:
        mean += ind.responses[DV_index]
    mean = mean / len(individuals)

    variance = 0 
    for ind in individuals:
        variance += (ind.responses[DV_index] - mean) * (ind.responses[DV_index] - mean)
    
    variance = variance / (len(individuals))
    return variance

def split_variance(splits_list, total_len, DV_index):
    """
    Calculates the variance of a split set of individuals with respect to a DV
    """
    split_variance_total = 0
    for list in splits_list:
        weight = len(list) / total_len
        split_variance_total += weight * variance(list,DV_index)
    return split_variance_total

def get_ordinal_meaining(codebook:Codebook,variable_index, number, is_less_than:bool):
    """
    Returns the meaning of an ordinal variable. Makes the results read a little bit better
    """
    significant_numbers = codebook.get_significant_numbers(variable_index)
    closest = 0
    min_difference = 10000
    for num in significant_numbers:
        if abs(num - number) < min_difference:
            closest = num
            min_difference = abs(num - number)
    if number == closest:
        if significant_numbers.index(num) == 0:
            if is_less_than:
                return codebook.get_meaning_of_number(variable_index,number)
            else:
                 return codebook.get_meaning_of_number(variable_index,significant_numbers[1] - 0.01)
        if significant_numbers[-1] == num:
            if is_less_than:
                return codebook.get_meaning_of_number(variable_index,significant_numbers[-2] + 0.01)
            else:
                 return codebook.get_meaning_of_number(variable_index,significant_numbers[-1])

    #less than side is closer
    if closest - number < 0:
        if is_less_than:
            return codebook.get_meaning_of_number(variable_index,number)
        else:
            return codebook.get_meaning_of_number(variable_index,significant_numbers[significant_numbers.index(closest) + 1] - 0.01)
    else:
        if is_less_than:
            return codebook.get_meaning_of_number(variable_index,significant_numbers[significant_numbers.index(closest) - 1] + 0.01)
        else:
            return codebook.get_meaning_of_number(variable_index,number)
    
def get_variance_splits(individuals,DV_index,IV_index,IV_type,codebook:Codebook):
    """
    Returns the propper split of a variable with respect to variance 
    """
    if 'Nominal' in IV_type or 'binary' in IV_type:
        return get_nominal_splits(individuals,IV_index)
    if 'Ordinal' in IV_type or 'Ratio' in IV_type:
        return get_ordinal_splits(individuals,DV_index,IV_index,codebook,variance_if_split_at)

def split_entropy(splits_list, total_len, DV_index):
    """
    Calculates the entropy of a split set of individuals with respect to a DV
    """
    split_entropy_total = 0
    for list in splits_list:
        weight = len(list) / total_len
        split_entropy_total += weight * entropy(list,DV_index)
    return split_entropy_total

def get_entropy_splits(individuals,DV_index,IV_index,IV_type,codebook:Codebook):
    """
    Returns the propper split of a variable with respect to entropy
    """
    if 'Nominal' in IV_type or 'binary' in IV_type:
        return get_nominal_splits(individuals,IV_index)
    if 'Ordinal' in IV_type or 'Ratio' in IV_type:
        return get_ordinal_splits(individuals,DV_index,IV_index,codebook,entropy_if_split_at)

def get_ordinal_splits(individuals,DV_index,IV_index,codebook:Codebook, split_function):
    lower = codebook.get_min_of_variable(IV_index)
    upper = codebook.get_max_of_variable(IV_index)
    #print("lower: " + str(lower))
    #print("upper: " + str(upper))
    best_split = -1
    best_split_result = 100000

    for i in range(int(lower), int(upper)+1):
        val = split_function(individuals,IV_index, DV_index, i)
        if val < best_split_result:
            best_split = i
            best_split_result = val
            #print("Best Val: ", end='')
        #print(val)

    list1 = list(filter(lambda ind: ind.responses[IV_index] < best_split,individuals))
    list2 = list(filter(lambda ind: ind.responses[IV_index] >= best_split,individuals))
    return [(best_split,list1),(best_split,list2)]

def get_nominal_splits(individuals,IV_index):
    """
    Splits a group of individuals by their response to a nominal question
    """
    #get all IV_Responses
    responses = set()
    for ind in individuals:
        if ind.responses[IV_index] not in responses:
            responses.add(ind.responses[IV_index])
    #make list of individuals for each of these splits
    splits = []
    for response in responses:
        splits.append((response, list(filter(lambda ind:ind.responses[IV_index] == response,individuals))))
    return splits

def variance_if_split_at(individuals, IV_index, DV_index,split):
    """
    Returns the variance of a group of individuals with respect to the DV if the IV variable is split into 2 groups:
    One group greater than the split
    One group less than the split
    """
    list1 = list(filter(lambda ind: ind.responses[IV_index] < split,individuals))
    list2 = list(filter(lambda ind: ind.responses[IV_index] >= split,individuals))
    return ((len(list1) / len(individuals)) * variance(list1,DV_index)) + ((len(list2) / len(individuals)) * variance(list2,DV_index)) 

def entropy_if_split_at(individuals, IV_index, DV_index,split):
    """
    Returns the variance of a group of individuals with respect to the DV if the IV variable is split into 2 groups:
    One group greater than the split
    One group less than the split
    """

    list1 = list(filter(lambda ind: ind.responses[IV_index] < split,individuals))
    list2 = list(filter(lambda ind: ind.responses[IV_index] >= split,individuals))

    list1_weighted_entropy = (len(list1) / len(individuals)) * entropy(list1,DV_index)
    list2_weighted_entropy = (len(list2) / len(individuals)) * entropy(list2,DV_index)

    return list1_weighted_entropy + list2_weighted_entropy

def entropy(individuals, DV_index):
    """
    Calculates Entropy
    """
    entropy = 0

    #lim x -> 0 of log x * x = 0
    if len(individuals) == 0:
        return 0
    
    #get DV Values
    DV_Values = set()
    for i in individuals:
        if i.responses[DV_index] not in DV_Values:
            DV_Values.add(i.responses[DV_index])
    
    #for all DV values, find prob_value * log ( prob_value)
    for value in DV_Values:
        value_len = len(list(filter(lambda indv: indv.responses[DV_index] == value, individuals)))
        prob_value = value_len / len(individuals)
        if(prob_value == 0):
            continue
        entropy += prob_value * math.log2(prob_value)
    entropy *= -1
    return entropy

def prune_tree(decision_tree,tuning_set,DV_index,DV_type):
    current_error = calculate_decision_tree_error_on_set(decision_tree,tuning_set,DV_index,DV_type)
    cuts = 0
    while True:
        nodes = get_all_non_leaf_nodes(decision_tree)
        if len(nodes) == 0:
            return decision_tree
        best_error = 9999999
        best_cut = nodes[-1]
        for node in nodes:
            error = get_error_if_turn_to_leaf(decision_tree,node,tuning_set, DV_index,DV_type)
            if error > best_error:
                best_cut = node
                best_error = error
        if best_error <= current_error:
            best_cut.is_leaf = True
            current_error = best_error
            cuts += 1
        else:
            break
"""
A much faster pruning method, although it may be more inaccurate
"""
def prune_fast(decision_tree,tuning_set,DV_index,DV_type):
    current_error = calculate_decision_tree_error_on_set(decision_tree,tuning_set,DV_index,DV_type)
    nodes = get_all_non_leaf_nodes(decision_tree)
    if len(nodes) == 0:
        return decision_tree
    for node in nodes[::-1]:
        error = get_error_if_turn_to_leaf(decision_tree,node,tuning_set, DV_index,DV_type)
        if error <= current_error:
            node.is_leaf = True
            current_error = error

def turn_node_to_leaf(node):
    node.is_leaf = True

def get_error_if_turn_to_leaf(decision_tree, node, tuning_set, DV_index,DV_type):
    node.is_leaf = True
    performance = calculate_decision_tree_error_on_set(decision_tree,tuning_set, DV_index,DV_type)
    
    node.is_leaf = False
    return performance
def get_all_non_leaf_nodes(decision_tree):
    """
    Returns all nodes that aren't leaves
    """
    if decision_tree.is_leaf:
        return []
    nodes = [decision_tree]
    for branch in decision_tree.children:
        new_nodes = get_all_non_leaf_nodes(branch[3])
        nodes.extend(new_nodes)
    
    return nodes


def calculate_decision_tree_error_on_set(decision_tree,tuning_set,DV_index,DV_type):
    sum = 0
    for i in tuning_set:
        sum += calculate_decision_tree_error_on_individual(decision_tree, i,DV_index,DV_type)
    return sum / len(tuning_set)

#checks if the decision returns the correct value for a given item, returns 1 if it does, 0 if not
def calculate_decision_tree_error_on_individual(decision_tree,individual,DV_index,DV_type):
    classification = decision_tree.classify(individual)
    if 'Nominal' in DV_type or 'binary' in DV_type:
        if classification == individual.responses[DV_index]:
            return 0
        else:
            return 1
    if 'Ordinal' in DV_type or 'Ratio' in DV_type:
        return (individual.responses[DV_index]- classification) * (individual.responses[DV_index] - classification)
#===============================================================================================================================================================
def training_set(individuals, every_num):
    """
        makes a traning set that includes every individual who's index is divisible by 'every_num'
    """
    training = []
    for i in range(len(individuals)):
        if not (i % every_num == 0):
            training.append(individuals[i])
    return training

def tuning_set(individuals,every_num):
    """
        makes a tuning set that picks every individual who's index is divisible by 'every_num'
    """
    tuning = []
    for i in range(len(individuals)):
        if (i % every_num == 0):
            tuning.append(individuals[i])
    return tuning

def make_individuals_list(individuals_list):
    """
        takes a list of voting histories and turns it into individuals
    """
    individuals = []
    for indv in individuals_list:
        ind = list(map(int,indv))
        individuals.append(Individual(ind))
    return individuals

def print_help():
    print("This program uses the format:")
    print("\tdecision_tree [Data File] [Codebook File] [DV_Name]")
    print('\tOptional: decision_tree [Data File] [Codebook File] [DV_Name] [IV_Name] [IV_Name] ... ')
    print('\t\t - restricts analysis to only the IVs you list in the command')


def parse_command_line(command_line) -> None:
    """
        Reads command line, makes tree and then displays it based on inputs
    """
    if '-h' in command_line:
        print_help()
        return
    if len(command_line) > 3:
        variables = list(map(lambda x: str(x),command_line[3:]))
    try:
        file = command_line[0] #Location of the data in excel format
        codebook_file = command_line[1] #location of the codebook
        DV = command_line[2] #Name of the dependent_variable
    except(IndexError):
        raise IndexError("Please provide input in the format decision_tree.py Dataset_File DV_Name Codebook_file Codebook_collumn_with_Variable_types_index")
    try:
        data = pd.ExcelFile(file) #data friom the excel file, raw pandas excel file
    except(FileNotFoundError):
        raise FileNotFoundError("Filename for Excel data sheet isn't valid, please input the correct file")
    individuals_list = data.parse().to_numpy() #list of individuals in numpy format, basically list of the rows of the data
    codebook = Codebook(codebook_file)
    individuals = make_individuals_list(individuals_list)

    #removes all irrlevant IVs
    if len(command_line) > 3:
        num_removed = 0
        for i in range(len(codebook.descriptions)):
            if (codebook.get_dataprac_number(i - num_removed) not in variables) and not (DV == codebook.get_dataprac_number(i - num_removed)):
                codebook.descriptions.pop(i - num_removed)
                for ind in individuals:
                    ind.responses.pop(i - num_removed)
                num_removed += 1
    random.shuffle(individuals) #shuffles data order
    DV_index = codebook.get_index_of_dataprac_number(DV)
    DV_type = codebook.get_datatype_of_variable(DV_index)

    training = training_set(individuals,5)
    tuning = tuning_set(individuals,5)

    if 'Nominal' in DV_type or 'binary' in DV_type:
        tree = make_nominal_decision_tree(training,codebook,DV_index,individuals[0].responses[DV_index])

    if 'Ordinal' in DV_type or 'Ratio' in DV_type:
        tree = make_ordinal_decision_tree(training,codebook,DV_index,individuals[0].responses[DV_index])
    
    prune_tree(tree,tuning,DV_index,DV_type)
    tree.limit_depth(5) # limits the tree depth for added readability, can be deleted with no negative consequences
    tree.print_tree(0)
    return tree
    
    #print(tree.classify(individuals[0]))

#===============================================================================================================================================================
if __name__ == "__main__":
    parse_command_line(sys.argv[1:])