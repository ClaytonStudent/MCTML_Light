# -*- coding: utf-8 -*-
from tree import TreeNode

class MCT():
    def __init__(self, root = None):
        self.path = [] #  保存nodes
        self.nodes = [] # add nodes
        if root:
            self.root = root
        else:
            self.root = TreeNode(None,None,1.0)
    def add_node_to_path(self, node):
        self.path.append(node)
    
    def add_all_nodes_to_tree(self,node):
        children = list(node._children.values())
        for child in children:
            self.add_node_to_tree(child)
    
    def add_node_to_tree(self, node):
        self.nodes.append(node)
        
    def get_node_from_tree(id):
        return self.notes[id]
    
    def do_move(self, node):
        self.root = node
    
     