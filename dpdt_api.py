from flask import Flask, jsonify, request
import json
import decision_tree as dt

def tree_to_dict(tree:dt.Decision_Tree):
    tree_dict = {}
    tree_dict["classification"] = str(tree.classification)
    tree_dict["variable_description"] = str(tree.variable_description)
    tree_dict["classificaiton_meaning"] = str(tree.codebook.get_meaning_of_number(tree.DV_index,tree.classification))
    if tree.is_leaf:
        tree_dict["children"] = []
    else:
        tree_dict["children"] = list(map(lambda a: tree_to_dict(a[3]),tree.children))
    return tree_dict

def getTreeDict(IV:str, DVs:list = ['DP1', 'DP5']):
    tree = dt.parse_command_line(['DatapracRStudio.xlsx', 'DatapracCodebook.xlsx'] + [IV] + DVs)
    return tree_to_dict(tree)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def query_records():
    query = dict(request.args)
    print(query)
    try:
        IV = query['independent_variable']
        DVs = query['dependent_variables']
        return jsonify(getTreeDict(IV,DVs.split(',')))
    except Exception as e:
        print(e)
        return jsonify({'error': 'data not found'})
    
    

app.run()