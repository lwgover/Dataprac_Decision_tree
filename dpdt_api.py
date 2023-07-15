from flask import Flask, jsonify, request
import json
import decision_tree as dt

#(classification,classifier_description,meaning_of_split,sub_tree)
#print(str(branch[1]) +" " +  str(branch[2]) + ": ",end='')
def tree_to_dict(tree:dt.Decision_Tree,tree_dict = {}):
    tree_dict["classification"] = str(tree.classification)
    tree_dict["variable_description"] = str(tree.variable_description)
    tree_dict["n"] = len(tree.individuals)
    tree_dict["classificaiton_meaning"] = str(tree.codebook.get_meaning_of_number(tree.DV_index,tree.classification))
    if tree.is_leaf:
        tree_dict["children"] = []
    else:
        if tree.children[0][2] == 'closer to ':
            tree_dict["children"] = list(map(lambda a: tree_to_dict(a[3],{"classifier_description":a[1]}),tree.children))
        else:
            tree_dict["children"] = list(map(lambda a: tree_to_dict(a[3],{"classifier_description":a[2]}),tree.children))
    return tree_dict

def tree_to_data(tree:dt.Decision_Tree):
    data_dict = {}
    data_dict["min"] = tree.codebook.get_min_of_variable(tree.DV_index)# minimum value of DV
    data_dict["max"] = tree.codebook.get_max_of_variable(tree.DV_index)# maximum value of DV
    data_dict["meanings"] = tree.codebook.descriptions[tree.DV_index]["numbers_and_meanings"]#dict where each interesting number goes to the meaning of that number
    return data_dict

def getTreeDict(DV:str, IVs:list = ['DP1', 'DP5']):
    tree = dt.parse_command_line(['data/DatapracRStudio.xlsx', 'data/DatapracCodebook.xlsx'] + [DV] + IVs)
    final_dict = {}
    final_dict["tree"] = tree_to_dict(tree,{})
    final_dict["data"] = tree_to_data(tree)
    return final_dict

app = Flask(__name__)

@app.route('/tree/<name>',methods=['GET'])
def get_tree(name=None):
    if name is None:
        return jsonify({'error': 'Please provide data labels to make tree from'}) 
    print(name)
    query = name.split(',')
    try:
        DV = query[0]
        IVs = query[1:]

        resp = jsonify(getTreeDict(DV,IVs))  #Binds data to a response object  
        # Set the Access Control Allow heades  
        resp.headers['Access-Control-Allow-Origin'] = '*'   # Allows cross origin requests
        return resp # Return response with Access Control Allow Headers 

    except Exception as e:
        print(e)
        return jsonify({'error': 'data not found'})
    
@app.route('/info/',methods=['GET'])
def get_info():

    resp = jsonify(dt.get_codebook('data/DatapracCodebook.xlsx'))  #Binds data to a response object  
    # Set the Access Control Allow heades  
    resp.headers['Access-Control-Allow-Origin'] = '*'   # Allows cross origin requests
    return resp

if __name__ == "__main__":
    app.run()