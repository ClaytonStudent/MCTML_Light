# -*- coding: utf-8 -*-
from tree import TreeNode

class MCT():
    def __init__(self, root = None):
        self.nodes = [] # add nodes
    
    def add_all_nodes_to_tree(self,node):
        children = list(node._children.values())
        for child in children:
            self.add_node_to_tree(child)
    
    def add_node_to_tree(self, node):
        self.nodes.append(node)
        
    def get_node_from_tree(id):
        return self.notes[id]
     