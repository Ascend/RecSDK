import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


class GraphPartitioner:
    def __init__(self):
        self.signature_def = None
        self.graph = None
        self.op_node_lookup = dict()
        self.input_op_nodes = []
        self.output_op_nodes = []
        self.tensor_node_lookup = dict()
        self.heavy_load_ops = []
        self.embedding_lookup_op_type = None
        self.first_heavy_load_on_sparse_path = set()
        self.first_op_after_lookup = []
        self.seen = set()
        self.post_out = set()
        self.partition_to_first_heavy_load = False

        self.sparse_lookup_ops = []
        self.sparse_lookup_tensors = []
        self.input_nodes = []
        self.output_nodes = []

    @staticmethod
    def has_gray_downstreams(op):
        gray_list = ["DynamicPartition"]
        down_ops = ge.get_forward_walk_ops([op])
        for op in down_ops:
            if op.type in gray_list:
                return True
        return False

    def set_embedding_lookup_op_type(self, s):
        self.embedding_lookup_op_type = s

    def get_sub_graph(self):
        for op in self.graph.get_operations():
            if self._is_embedding_lookup(op):
                self.sparse_lookup_ops.append(op)
        if not self.sparse_lookup_ops:
            for op in self.graph.get_operations():
                is_top_op = True
                for op1 in self.graph.get_operations():
                    for tensor in op1.outputs:
                        if tensor in op.inputs:
                            is_top_op = False
                            break
                    if not is_top_op:
                        break
                if is_top_op:
                    self.sparse_lookup_ops.append(op)
        check_ops = self.sparse_lookup_ops
        self.sparse_lookup_ops = []
        for op in check_ops:
            if not self.has_gray_downstreams(op):
                self.sparse_lookup_ops.append(op)
                self.sparse_lookup_tensors.extend(op.outputs)

        for op in self.graph.get_operations():
            for tensor in self.sparse_lookup_tensors:
                if tensor in op.inputs:
                    self.input_nodes.append(op)
        for k, v in self.signature_def.outputs.items():
            op_name = (
                str(v)
                .split("\n")[0]
                .replace(" ", "")
                .replace('"', "")
                .split(":")[1]
                .split(":")[0]
            )
            for op in self.graph.get_operations():
                if op.name == op_name:
                    self.output_nodes.append(op)

        float_ups = []
        to_expand = []
        in_str = []

        for op in self.input_nodes:
            if op.type not in float_ups:
                if op.name not in in_str:
                    in_str.append(op.name)
            else:
                to_expand.append(op)

        while to_expand:
            candidates = []
            for top in to_expand:
                for op in self.graph.get_operations():
                    for tensor in op.inputs:
                        if tensor in top.outputs:
                            candidates.append(op)
            to_expand = []
            for op in candidates:
                if op.type not in float_ups:
                    if op.name not in in_str:
                        in_str.append(op.name)
                else:
                    to_expand.append(op)
        return str(in_str), str([op.name for op in self.output_nodes])

    def _is_embedding_lookup(self, op):
        if op.type in self.embedding_lookup_op_type:
            return True

        return False

    def _check_op_status(self):
        unseen_list = []
        for name, op_node in self.op_node_lookup.items():
            if not op_node.seen:
                unseen_list.append(name)
        return unseen_list
