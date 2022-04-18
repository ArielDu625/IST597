import os
import random
import time
import ast
import json
import pickle
from collections import deque

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset
from binarytree import Node
from graphviz import Digraph


OPERATORS = {
    'Add': '+',
    'Sub': '-',
    'USub': '-',
    'Mult': '*',
    'Div': '/',
    'Pow': '**',
    'Eq': '=='
}

functionOneInp = ['sin', 'cos', 'csc', 'sec', 'tan', 'cot',
				  'asin', 'acos', 'acsc', 'asec', 'atan', 'acot',
				  'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
				  'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
				  'exp']


class TreeNode:
    def __init__(self, node_idx, node_symbol, left_child=None, right_child=None):
        self.node_idx = node_idx
        self.node_symbol = node_symbol
        self.left_child = left_child
        self.right_child = right_child
        self.left_processed = False
        self.right_processed = False


class BinaryTree:
    def __init__(self, preorder_nodeIdx, preorder_symbol):
        assert len(preorder_nodeIdx) == len(preorder_symbol)
        assert len(preorder_nodeIdx) > 0

        self.root = TreeNode(int(preorder_nodeIdx[0]), preorder_symbol[0])
        stack = [self.root]
        n = len(preorder_nodeIdx)
        i = 1

        while i < n:
            stack_top = len(stack) - 1
            if preorder_nodeIdx[i] == '#' and (not stack[stack_top].left_processed):
                left = None
                stack[stack_top].left_child = left
                stack[stack_top].left_processed = True
            elif preorder_nodeIdx[i] == '#' and stack[stack_top].left_processed:
                right = None
                stack[stack_top].right_child = right
                stack[stack_top].right_processed = True
                stack.pop()
            elif preorder_nodeIdx[i] != '#' and (not stack[stack_top].left_processed):
                left = TreeNode(int(preorder_nodeIdx[i]), preorder_symbol[i])
                stack[stack_top].left_child = left
                stack[stack_top].left_processed = True
                stack.append(left)
            else:
                right = TreeNode(int(preorder_nodeIdx[i]), preorder_symbol[i])
                stack[stack_top].right_child = right
                stack[stack_top].right_processed = True
                stack.pop()
                stack.append(right)

            i += 1
        
        # for node whose left child is not None and right right is None, change the left subtree to it's right side
        queue = deque([self.root])
        while queue:
            cur_node = queue.popleft()
            if cur_node.left_child is not None and cur_node.right_child is None:
                cur_node.right_child = cur_node.left_child
                cur_node.left_child = None
            
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)

    def preorder(self, node, printSymbol=False):
        if node is None:
            return
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')
        self.preorder(node.left_child, printSymbol=printSymbol)
        self.preorder(node.right_child, printSymbol=printSymbol)

    def inorder(self, node, printSymbol=False):
        if node is None:
            return
        self.inorder(node.left_child, printSymbol=printSymbol)
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')
        self.inorder(node.right_child, printSymbol=printSymbol)

    def postorder(self, node, printSymbol=False):
        if node is None:
            return
        self.postorder(node.left_child, printSymbol=printSymbol)
        self.postorder(node.right_child, printSymbol=printSymbol)
        if printSymbol:
            print(node.node_symbol, end=' ')
        else:
            print(node.node_idx, end=' ')
    
    def _expression(self, root, expr):
        if root is None:
            return

        expr.append('(')
        self._expression(root.left_child, expr)

        cur = root.node_symbol
        if cur.startswith('-'):
            expr.append('-')
            cur = cur[1:]
        if '/' in cur:
            numerator, denominator = cur.split('/')
            expr.append(numerator)
            expr.append('/')
            expr.append(denominator)
        else:
            expr.append(cur)

        self._expression(root.right_child, expr)
        expr.append(')')

    def expression(self):
        expr = []
        self._expression(self.root, expr)
        return expr

    def levelorder_symbol(self):
        """
        return a list of symbols with the same order as node id
        :return: list
        """
        symbols = []
        queue = deque([self.root])
        while queue:
            cur = queue.popleft()
            symbols.append(cur.node_symbol)
            if cur.left_child:
                queue.append(cur.left_child)
            if cur.right_child:
                queue.append(cur.right_child)
        return symbols


class EquationTree:

    def __init__(self, evars, numNodes, evariables, depth, nodeNum, func, label):
        self.evars = evars.split(',')
        self.numNodes = int(numNodes)
        self.evariables = {}
        for var_name, var_idx in evariables.items():
            self.evariables[var_idx] = var_name
        self.nodeDepth = depth.split(',')
        self.depth = 0
        for d in self.nodeDepth:
            if d != '#':
                self.depth = max(self.depth, int(d))

        self.nodeNum = nodeNum.split(',')
        self.func = func.split(',')
        self.label = int(label)

        assert len(self.evars) == len(self.nodeDepth)
        assert len(self.nodeDepth) == len(self.nodeNum)
        assert len(self.nodeNum) == len(self.func)

        self.node_list = []
        # COO format edge index
        self.edge_list = [[], []]

        self._build()

    def _build(self):
        node_idx = []
        node_symbol = []
        for i in range(len(self.func)):
            node_idx.append(self.nodeNum[i])

            if self.func[i] == 'Symbol':
                var_idx = int(self.evars[i].split('_')[1])
                var_name = self.evariables[var_idx]
                node_symbol.append(var_name)
            elif self.func[i] == "NegativeOne":
                node_symbol.append("-1")
            elif self.func[i] == "Pi":
                node_symbol.append(self.evars[i])
            elif self.func[i] == "One":
                node_symbol.append("1")
            elif self.func[i] == "Half":
                node_symbol.append("1/2")
            elif self.func[i] == 'Integer':
                node_symbol.append(self.evars[i])
            elif self.func[i] == "Rational":
                node_symbol.append(self.evars[i])
            elif self.func[i] == "Float":
                node_symbol.append(self.evars[i])
            elif self.func[i] == "Equality":
                node_symbol.append("==")
            elif self.func[i] == 'Mul':
                node_symbol.append('*')
            elif self.func[i] == 'Add':
                node_symbol.append('+')
            elif self.func[i] == 'Pow':
                node_symbol.append('**')
            else:
                node_symbol.append(self.func[i])

        self.bt = BinaryTree(node_idx, node_symbol)
        
        
        # node symbol in the same order as node id
        node_queue = deque([self.bt.root])
        while node_queue:
            cur_node = node_queue.popleft()
            self.node_list.append(cur_node.node_symbol)
            if cur_node.left_child:
                s_node_idx, e_node_idx = cur_node.left_child.node_idx, cur_node.node_idx
                self.edge_list[0].append(s_node_idx)
                self.edge_list[1].append(e_node_idx)
                # AST undirected edge
                self.edge_list[0].append(e_node_idx)
                self.edge_list[1].append(s_node_idx)

                node_queue.append(cur_node.left_child)
            if cur_node.right_child:
                s_node_idx, e_node_idx = cur_node.right_child.node_idx, cur_node.node_idx
                self.edge_list[0].append(s_node_idx)
                self.edge_list[1].append(e_node_idx)
                # AST undirected edge
                self.edge_list[0].append(e_node_idx)
                self.edge_list[1].append(s_node_idx)
                
                node_queue.append(cur_node.right_child)

    def get_result(self):
        return self.node_list, self.edge_list

    def get_label(self):
        return self.label

    def get_depth(self):
        return self.depth

    def get_symbols(self):
        return self.bt.levelorder_symbol()
    
    def get_expression(self):
        expr_list = self.bt.expression()
        return expr_list


class ExprVisit(ast.NodeTransformer):
    '''
    Parsing an expression and generating the AST by the post-order traversal.
    For each expression, its variables must be a single character.
    '''
    def __init__(self, is_dag=True):
        self.is_dag = is_dag
        self.node_list = []
        self.edge_list = []
        self.subtree_memo = []

    def merge_node(self, node):
        '''Determine whether the current node can be merged'''
        same_node = None
        if self.subtree_memo is None:
            return same_node
        cur_node_type = 'binop' if hasattr(node, 'left') else 'unaryop'
        for p_node in self.subtree_memo:
            cur_p_node_type = 'binop' if hasattr(p_node, 'left') else 'unaryop'
            if cur_p_node_type != cur_node_type:
                continue
            elif cur_p_node_type == cur_node_type == 'binop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.left) == ast.dump(node.left) \
                    and ast.dump(p_node.right) == ast.dump(node.right):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)
            elif cur_p_node_type == cur_node_type == 'unaryop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.operand) == ast.dump(node.operand):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)

        return same_node
    def visit_Compare(self, node):
        """
        Scan compare operator: ==
        """
        self.generic_visit(node)
        node_str = ast.dump(node.ops[0]) + str(node.col_offset)
        self.node_list.append(node_str)

        return node

    def visit_Call(self, node):
        """
        Scan function operators, such as cos, sin, tan,...
        """
        self.generic_visit(node)
        node_str = node.func.id + '()' + str(node.col_offset)

        self.node_list.append(node_str)
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'ops'):
            node_parent_str = ast.dump(node.parent.ops[0]) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'func'):
            node_parent_str = node.parent.func.id + '()' + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_BinOp(self, node):
        '''
        Scan binary operators, such as +, -, *, &, |, ^
        '''
        self.generic_visit(node)
        # node.col_offset is unique identifer
        node_str = ast.dump(node.op) + str(node.col_offset)
        if self.is_dag:
            same_node = self.merge_node(node)
            if same_node == None:
                self.subtree_memo.append(node)
                self.node_list.append(node_str)
            else:
                for idx in range(len(self.edge_list) - 1, -1, -1):
                    if node_str == self.edge_list[idx][0] or \
                        node_str == self.edge_list[idx][1]:
                        del self.edge_list[idx]
                node_str = same_node
        else:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'ops'):
            node_parent_str = ast.dump(node.parent.ops[0]) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'func'):
            node_parent_str = node.parent.func.id + '()' + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_UnaryOp(self, node):
        '''
        Scan unary operators, such as - and ~
        '''
        self.generic_visit(node)
        node_str = ast.dump(node.op) + str(node.col_offset)
        
        if self.is_dag:
            same_node = self.merge_node(node)
            if same_node == None:
                self.subtree_memo.append(node)
                self.node_list.append(node_str)
            else:
                for idx in range(len(self.edge_list) - 1, -1, -1):
                    if node_str == self.edge_list[idx][0] \
                        or node_str == self.edge_list[idx][1]:
                        del self.edge_list[idx]
                node_str = same_node
        else:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'ops'):
            node_parent_str = ast.dump(node.parent.ops[0]) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'func'):
            node_parent_str = node.parent.func.id + '()' + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_Name(self, node):
        '''
        Scan variables
        '''
        self.generic_visit(node)
        # node.col_offset will allocate a unique ID to each node
        node_str = node.id
        if node.id in functionOneInp:
            return node

        if not self.is_dag:
            node_str += '()' + str(node.col_offset)
        if node_str not in self.node_list and node_str not in functionOneInp:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'ops'):
            node_parent_str = ast.dump(node.parent.ops[0]) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'func'):
            node_parent_str = node.parent.func.id + '()' + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    # Use visit_Constant when python version >= 3.8
    # def visit_Constant(self, node):
    def visit_Num(self, node):
        '''
        Scan numbers
        '''
        self.generic_visit(node)
        node_str = str(node.n)
        if not self.is_dag:
            node_str += '()' + str(node.col_offset)
        if node_str not in self.node_list:
            self.node_list.append(node_str)

        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'ops'):
            node_parent_str = ast.dump(node.parent.ops[0]) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        elif hasattr(node.parent, 'func'):
            node_parent_str = node.parent.func.id + '()' + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])
        
        # node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
        # self.edge_list.append([node_str, node_parent_str])

        return node

    def get_result(self):
        return self.node_list, self.edge_list


def expr2graph(expr, is_dag=True):
    '''
    Convert a expression to a MSAT graph.

    Parameters:
        expr: A string-type expression.

    Return:
        node_list:  List for nodes in graph.
        edge_list:  List for edges in graph.
    '''
    ast_obj = ast.parse(expr)
    for node in ast.walk(ast_obj):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    vistor = ExprVisit(is_dag=is_dag)
    vistor.visit(ast_obj)
    node_list, edge_list = vistor.get_result()

    return node_list, edge_list


def dot_expr(nodes, edges):
    """
    Dot a graph G with graphviz.
    Parameters:
        nodes:  List, each element represents a node in G.
        edges:  List, each element is a two-tuples representing an directed edge in G.
    Returns:
        dot:    A Digraph object.
    """
    dot = Digraph()

    for node in nodes:
        label = node.split('()')[0]
        if label in OPERATORS:
            label = OPERATORS[label]
        dot.node(node, label)

    for edge in edges:
        dot.edge(edge[0], edge[1])

    dot.render(filename=str(time.time()), format='pdf')


def load_single_equation(example):
    """
    :param example: a dictionary of schema
        {
            "equation": {
                "vars": value for each constant node 'NegativeOne', 'Pi', 'One',
                        'Half', 'Integer', 'Rational', 'Float'
                "numNodes": number of nodes in this tree, discounting #
                "variables": dictionary of ?,
                "depth": depth of each node in this tree
                "nodeNum": unique ids of each node
                "func": the actual list of nodes in this (binary) equation tree,
                    unary functions are still encoded as having two children,
                    the right one being NULL (#)
            },
            "label": "1" if the lhs of the equation equals rhs else "0"
        }
    :return: An EquationTree corresponding to 'example', paired with it's depth and it's label
    """
    evars = example['equation']['vars']
    numNodes = example['equation']['numNodes']
    variables = example['equation']['variables']
    depth = example['equation']['depth']
    nodeNum = example['equation']['nodeNum']
    func = example['equation']['func']
    label = example['label']

    et = EquationTree(evars, numNodes, variables, depth, nodeNum, func, label)
    return et, et.get_depth(), et.get_label()


def build_equation_tree_examples_list(train_jsonfile, test_jsonfile=None, val_jsonfile=None, depth=None):
    """

    :param train_jsonfile:
    :param test_jsonfile:
    :param val_jsonfile:
    :param depth: if depth is given, only return the equation tree of the given depth
    :return: a list of trio (BinaryTree, depth, label), a dict of all symbols
    """
    train_trios = []
    test_trios = []
    val_trios = []
    symbol_dict = {}
    dag_symbol_dict = {}

    with open(train_jsonfile, 'rt') as f:
        group_list = json.loads(f.read())

    for i, group in enumerate(group_list):
        for example in group:
            et, d, label = load_single_equation(example)
            if depth is not None and d not in depth:
                continue
            train_trios.append((et, d, label))
            et_symbols = et.get_symbols()
            for s in et_symbols:
                if s not in symbol_dict:
                    symbol_dict[s] = len(symbol_dict)

            expr_list = et.get_expression()
            for c in expr_list:
                if c != '(' and c != ')' and c not in dag_symbol_dict:
                    dag_symbol_dict[c] = len(dag_symbol_dict)

    if test_jsonfile is not None:
        with open(test_jsonfile, 'rt') as f:
            test_groups = json.loads(f.read())

        for i, group in enumerate(test_groups):
            for example in group:
                et, d, label = load_single_equation(example)
                if depth is not None and d not in depth:
                    continue
                test_trios.append((et, d, label))
                et_symbols = et.get_symbols()
                for s in et_symbols:
                    if s not in symbol_dict:
                        symbol_dict[s] = len(symbol_dict)

                expr_list = et.get_expression()
                for c in expr_list:
                    if c != '(' and c != ')' and c not in dag_symbol_dict:
                        dag_symbol_dict[c] = len(dag_symbol_dict)

    if val_jsonfile is not None:
        with open(val_jsonfile, 'rt') as f:
            val_groups = json.loads(f.read())

        for i, group in enumerate(val_groups):
            for example in group:
                et, d, label = load_single_equation(example)
                if depth is not None and d not in depth:
                    continue
                val_trios.append((et, d, label))
                et_symbols = et.get_symbols()
                for s in et_symbols:
                    if s not in symbol_dict:
                        symbol_dict[s] = len(symbol_dict)
                
                expr_list = et.get_expression()
                for c in expr_list:
                    if c != '(' and c != ')' and c not in dag_symbol_dict:
                        dag_symbol_dict[c] = len(dag_symbol_dict)

    return symbol_dict, dag_symbol_dict, train_trios, test_trios, val_trios


class GraphExprDataset(InMemoryDataset):
    """
    Construct the dataset we will use when training the model
    """
    def __init__(self, root, train_filename, test_filename, val_filename, is_dag=False, valid_depth=None):
        self.symbol_vocab = None
        self.train_filename = f'{root}/raw/{train_filename}'
        self.test_filename = f'{root}/raw/{test_filename}'
        self.val_filename = f'{root}/raw/{val_filename}'
        self.train_size = 0
        self.test_size = 0
        self.val_size = 0
        self.is_dag = is_dag
        self.valid_depth = valid_depth
        self.dag_symbol_vocab = None

        # self.graph_node_list = []
        # self.graph_edge_list = []

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = [self.train_filename, self.test_filename, self.val_filename]
        return files
        
    @property
    def processed_file_names(self):
        if self.is_dag:
            return ['dag.pt']
        else:
            return ['ast.pt']

    def download(self):
        pass

    def _generate_ast(self, equation_tree: EquationTree):
        """

        :param equation_tree:
        :return:
        """
        node_list, edge_list = equation_tree.get_result()

        node_feature = []
        for node_symbol in node_list:
            feature = [0] * len(self.symbol_vocab)
            feature[self.symbol_vocab[node_symbol]] = 1

            node_feature.append(feature)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long)

        return x, edge_index
    
    def _generate_dag(self, equation_tree: EquationTree):
        expr = ' '.join(equation_tree.get_expression())
        try:
            node_list, edge_list = expr2graph(expr)
            # self.graph_node_list.append(node_list)
            # self.graph_edge_list.append(edge_list)

        except:
            raise ValueError(expr)
        
        node_feature = []
        for node in node_list:
            tag = node.split('()')[0]
            if tag in OPERATORS: 
                tag = OPERATORS[tag]

            feature = [0] * len(self.dag_symbol_vocab)
            feature[self.dag_symbol_vocab[tag]] = 1
            node_feature.append(feature)
        
        COO_edge_idx = [[], []]
        for edge in edge_list:
            s_node, e_node = node_list.index(edge[0]), node_list.index(edge[1])
            COO_edge_idx[0].append(s_node), COO_edge_idx[1].append(e_node)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(COO_edge_idx, dtype=torch.long)

        return x, edge_index

    def process(self):
        data_list = []

        symbol_dict, dag_symbol_dict, train_trios, test_trios, val_trios = build_equation_tree_examples_list(self.train_filename, 
                                                                                            self.test_filename, 
                                                                                            self.val_filename,
                                                                                            depth = self.valid_depth)
        self.symbol_vocab = symbol_dict
        self.dag_symbol_vocab = dag_symbol_dict
        trio_list = train_trios + test_trios + val_trios
        self.train_size = len(train_trios)
        self.test_size = len(test_trios)
        self.val_size = len(val_trios)

        for i, trio in tqdm(enumerate(trio_list)):
            et, d, label = trio

            if self.is_dag:
                x, edge_index = self._generate_dag(et)
            else:
                x, edge_index = self._generate_ast(et)
            
            y = [label]
            y = torch.tensor(y, dtype=torch.long)

            # feed the graph and label into Data
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # with open('graph_test.pkl', 'wb') as f:
        #     pickle.dump([self.graph_node_list, self.graph_edge_list], f)
        
        # pad the dataset TODO: check here if it's needed to pad the dataset( I guess it's no need to pad)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def test_equation_tree(et):
    bt = et.bt
    check_root = Node(str(bt.root.node_idx) + ":" + bt.root.node_symbol)
    stack = [(bt.root, check_root)]

    while stack:
        cur, check_cur = stack.pop()
        if cur.left_child:
            check_left = Node(str(cur.left_child.node_idx) + ":" + cur.left_child.node_symbol)
            check_cur.left = check_left
            stack.append((cur.left_child, check_left))
        if cur.right_child:
            check_right = Node(str(cur.right_child.node_idx) + ":" + cur.right_child.node_symbol)
            check_cur.right = check_right
            stack.append((cur.right_child, check_right))
    
    print(check_root)
    print(check_root.inorder)


if __name__ == "__main__":
    # testing the EquationTree class
    # randomly select a sample from each depth group and check the binary tree
    train_jsonfile = './dataset/raw/40k_train.json'
    test_jsonfile = './dataset/raw/40k_test.json'
    val_jsonfile = './dataset/raw/40k_val_shallow.json'
    files = [train_jsonfile, test_jsonfile, val_jsonfile]

    for fidx, jsonfile in enumerate(files):
        print(f"sampling from {jsonfile}...")
        selected = []

        with open(jsonfile, 'rt') as f:
            group_list = json.loads(f.read())

        for i, group in enumerate(group_list):
            group_size = len(group)
            if group_size == 0:
                continue

            sample_idx = random.randint(0, group_size-1)
            example = group[sample_idx]
            selected.append([example])

            evars = example['equation']['vars']
            numNodes = example['equation']['numNodes']
            variables = example['equation']['variables']
            depth = example['equation']['depth']
            nodeNum = example['equation']['nodeNum']
            func = example['equation']['func']
            label = example['label']
        
            et = EquationTree(evars, numNodes, variables, depth, nodeNum, func, label)

            if fidx == 0:
                print(f"sample from group with tree depth={i+1}:")
                print(f"depth: {depth}")
                print(f"nodeNum: {nodeNum}")
                print(f"func: {func}")
                print(f"vars: {evars}")
                print(f"variables: {variables}")
                print(f"label: {label}")
                print(f"levelorder symbol:{et.get_symbols()}")
                test_equation_tree(et)
                print(f"expression: {' '.join(et.get_expression())}")
                print()

        print(f"selected {len(selected)} trees...")

        filename = 'sampled_' + jsonfile.split('/')[-1]
        with open(f'./test/raw/{filename}', 'w') as f:
            json.dump(selected, f, indent=4)
    

    # testing the GraphExprDataset
    # construct dataset from sampled examples, then check the edge_index and label
    os.system(f'rm -rf test/processed/*')
    dataset = GraphExprDataset('test', 'sampled_40k_train.json', 'sampled_40k_test.json', 'sampled_40k_val_shallow.json', is_dag=True)
    print(f'train size: {dataset.train_size}')
    print(f'test size: {dataset.test_size}')
    print(f'val size: {dataset.val_size}')
    print(f'Symbol vocab: {dataset.symbol_vocab}')
    print(f'Dag symbol vocab: {dataset.dag_symbol_vocab}')

    train_set = dataset[:dataset.train_size]
    test_set = dataset[dataset.train_size: (dataset.train_size+dataset.test_size)]
    valid_set = dataset[(-dataset.val_size):]

    for sample in train_set:
        print(sample)
        print(sample.x)
        print(sample.edge_index)
        print(sample.y)
        print()

