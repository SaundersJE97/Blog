leaf_nodes = 0
def fill_tree(depth=8):
    global leaf_nodes
    if depth == 1:
        leaf_nodes += 1
        return {'state' : [],
                'leaf' : True}

    else:
        action0 = fill_tree(depth=depth-1)
        action1 = fill_tree(depth=depth-1)
        action2 = fill_tree(depth=depth-1)
    return {'state' : [],
            0 : action0,
            1 : action1,
            2 : action2,
            'leaf' : False}

tree = fill_tree(8)



print(tree)


