#include <cstring>
#include <iostream>
#include <set>

#include "all_ops.h"
#include "ge_ir_build.h"
#include "register/register_custom_pass.h"

using namespace ge;

namespace {
constexpr const char* const kTypeSub = "Sub";
constexpr const char* const kTypeAdd = "Add";
constexpr const char* const kTypeSqrt = "Sqrt";
constexpr const char* const kTypeRealDiv = "RealDiv";
constexpr const char* const kTypeBatchNorm = "BatchNorm";
constexpr const char* const kTypeReshape = "Reshape";
constexpr const char* const kTypeBatchMatmulV2 = "BatchMatMulV2";
constexpr const char* const kTypeBatchMatmulV3 = "BatchMatMulV3";
constexpr const char* const kTypeMatmulV2 = "MatMulV2";
constexpr const char* const kTypeMatmulV3 = "MatMulV3";
constexpr const char* const kTypePack = "Pack";
constexpr const char* const kTypeTranspose = "Transpose";
constexpr const char* const kTypeSoftmax = "SoftmaxV2";
constexpr const char* const kTypeMul = "Mul";
constexpr const char* const kTypeReduceSumD = "ReduceSumD";
constexpr const char* const kTypeTile = "Tile";
constexpr const char* const reshapeShapeName = "shape";
constexpr const char* const reshapeInputName = "x";
constexpr const char* const transposePermName = "perm";
constexpr const char* const transposeInputName = "x";
constexpr const char* const kTypeConst = "Const";
constexpr const char* const kTypeConstant = "Constant";
constexpr const char* const kTypeSqueeze = "Squeeze";
}  // namespace

void FindNodesCanFusion(GraphPtr& graph, std::vector<GNode>& mm_nodes)
{
    for (const GNode& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeMatmulV2) {
            mm_nodes.emplace_back(node);
        }
    }
}

graphStatus GetConstTransposeAttr(std::shared_ptr<GNode> mm_node, int32_t data_idx, bool& transpose_const)
{
    if (data_idx == 0) {
        if (mm_node->GetAttr(AscendString("transpose_x2"), transpose_const) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    } else {
        if (mm_node->GetAttr(AscendString("transpose_x1"), transpose_const) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

graphStatus GetCombinedDim(GNodePtr mm_node, int32_t data_idx, int32_t& combined_dim)
{
    int32_t const_idx = 1 - data_idx;
    TensorDesc const_desc;
    Shape const_shape;
    if (mm_node->GetInputDesc(const_idx, const_desc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const_shape = const_desc.GetShape();

    bool transpose_const;
    if (GetConstTransposeAttr(mm_node, data_idx, transpose_const) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if ((transpose_const != (data_idx == 1))) {
        combined_dim = const_shape.GetDim(0);
    } else {
        combined_dim = const_shape.GetDim(1);
    }

    return GRAPH_SUCCESS;
}

graphStatus GetConstTensor(GNodePtr mm_node, int32_t data_idx, Tensor& tensor)
{
    int32_t const_idx = 1 - data_idx;

    AscendString mm_name;
    if (mm_node->GetName(mm_name) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    mm_node->GetInputConstData(const_idx, tensor);

    return GRAPH_SUCCESS;
}

struct MmoeMatmulPassContext {
    struct MatmulNodeContext {
        GNodePtr node;
        std::shared_ptr<Tensor> tensor_ptr;

        MatmulNodeContext(GNodePtr n, std::shared_ptr<Tensor> t) : node(std::move(n)), tensor_ptr(std::move(t)) {};
    };
    std::vector<GNodePtr> middle_tensors;
    Shape root_shape;
    Shape cur_data_shape;
    int32_t root_output_idx;
    int32_t data_idx;
    int32_t const_idx;
    bool transpose_x1;
    bool transpose_x2;
    int32_t combined_dim;
    AscendString base_node_name;
    std::vector<MatmulNodeContext> matmul_node_context;

    MmoeMatmulPassContext(Shape root_shape, int32_t root_output_idx, int32_t data_idx, int32_t const_idx,
                          bool transpose_x1, bool transpose_x2, int32_t combined_dim, Shape cur_data_shape,
                          AscendString base_node_name)
        : root_shape(root_shape),
          root_output_idx(root_output_idx),
          data_idx(data_idx),
          const_idx(const_idx),
          transpose_x1(transpose_x1),
          transpose_x2(transpose_x2),
          combined_dim(combined_dim),
          cur_data_shape(cur_data_shape),
          base_node_name(base_node_name)
    {
    }

    graphStatus AddNode(GNodePtr n, int32_t data_idx)
    {
        auto tensor_ptr = std::make_shared<Tensor>();
        if (GetConstTensor(n, data_idx, *tensor_ptr) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        matmul_node_context.emplace_back(n, tensor_ptr);
        return GRAPH_SUCCESS;
    }

    void AddMiddleNode(GNodePtr node)
    {
        middle_tensors.emplace_back(std::move(node));
    }

    void AddMiddleNode(std::vector<GNodePtr> nodes)
    {
        for (auto node : nodes) {
            middle_tensors.emplace_back(std::move(node));
        }
    }
};

bool CheckConstType(AscendString node_type)
{
    return node_type == kTypeConst || node_type == kTypeConstant;
}

template <typename T>
graphStatus concat_matrics(const MmoeMatmulPassContext fusion_context, size_t k_dim, int32_t concat_dim,
                           uint8_t*& output_matric)
{
    if (concat_dim != 0 && concat_dim != 1) {
        return GRAPH_FAILED;
    }

    int32_t out_combined_dim = fusion_context.combined_dim * fusion_context.matmul_node_context.size();
    int32_t combined_dim = fusion_context.combined_dim;

    size_t elem_size = sizeof(T);
    const size_t total_elements = k_dim * out_combined_dim;
    const size_t total_bytes = total_elements * elem_size;

    uint8_t* result_bytes = new uint8_t[total_bytes];

    T* result = static_cast<T*>(static_cast<void*>(result_bytes));
    size_t current_offset = 0;
    for (const auto& rel : fusion_context.matmul_node_context) {
        uint8_t* data = rel.tensor_ptr->GetData();
        const T* src_data = static_cast<T*>(static_cast<void*>(data));

        if (concat_dim == 0) {
            uint8_t* dest = result_bytes + current_offset;
            std::memcpy(dest, data, k_dim * combined_dim * elem_size);
            current_offset += combined_dim * k_dim * elem_size;
        } else {
            for (size_t k = 0; k < k_dim; ++k) {
                uint8_t* src = data + k * combined_dim * elem_size;
                uint8_t* dest = result_bytes + current_offset + k * out_combined_dim * elem_size;
                std::memcpy(dest, src, combined_dim * elem_size);
            }
            current_offset += combined_dim * elem_size;
        }
    }

    output_matric = result_bytes;
    return GRAPH_SUCCESS;
}

bool CompareShape(Shape shape1, Shape shape2)
{
    int32_t dims1 = shape1.GetDimNum();
    int32_t dims2 = shape2.GetDimNum();
    if (dims1 != dims2) {
        return false;
    }

    for (int32_t i = 0; i < dims1; i++) {
        int32_t dim1 = shape1.GetDim(i);
        int32_t dim2 = shape2.GetDim(i);
        if (dim1 != dim2) {
            return false;
        }
    }

    return true;
}

graphStatus FindRootNode(GNode& cur_node, GNode& root_node, std::vector<GNodePtr>& middle_nodes,
                         int32_t& root_output_idx)
{
    AscendString cur_node_type;
    cur_node.GetType(cur_node_type);
    AscendString cur_name;
    cur_node.GetName(cur_name);
    if (cur_node_type != kTypeReshape && cur_node_type != kTypeSqueeze) {
        if (CheckConstType(cur_node_type)) {
            return GRAPH_FAILED;
        }
        root_node = cur_node;
        AscendString root_name;
        root_node.GetName(root_name);
        return GRAPH_SUCCESS;
    }

    int32_t input_size = cur_node.GetInputsSize();
    if (input_size <= 0) {
        return GRAPH_FAILED;
    }

    auto output_nodes = cur_node.GetOutDataNodesAndPortIndexs(0);
    if (output_nodes.size() > 1) {
        return GRAPH_FAILED;
    }

    int32_t non_const_input_num = 0;
    int32_t non_const_idx = 0;
    for (int32_t i = 0; i < input_size; i++) {
        auto input_node_pair = cur_node.GetInDataNodesAndPortIndexs(i);
        auto input_node = input_node_pair.first;
        AscendString input_type;
        input_node->GetType(input_type);
        if (CheckConstType(input_type)) {
            continue;
        }
        non_const_input_num++;
        non_const_idx = i;
    }

    if (non_const_input_num != 1) {
        return GRAPH_FAILED;
    }

    auto non_const_pair = cur_node.GetInDataNodesAndPortIndexs(non_const_idx);
    auto non_const_node = non_const_pair.first;
    root_output_idx = non_const_pair.second;
    middle_nodes.emplace_back(std::make_shared<GNode>(cur_node));
    return FindRootNode(*non_const_node, root_node, middle_nodes, root_output_idx);
}

graphStatus FindSingleDownStreamNodes(GNodePtr& root_node, GNodePtr& out_node)
{
    size_t output_size = root_node->GetOutputsSize();
    if (output_size != 1) {
        return GRAPH_FAILED;
    }

    auto output_nodes = root_node->GetOutDataNodesAndPortIndexs(0);
    if (output_nodes.size() != 1) {
        return GRAPH_FAILED;
    }

    out_node = output_nodes[0].first;
    return GRAPH_SUCCESS;
}

graphStatus FindMatmulNodes(GNode& root_node, int32_t middle_nodes_count, MmoeMatmulPassContext& context)
{
    auto output_nodes = root_node.GetOutDataNodesAndPortIndexs(context.root_output_idx);
    for (auto output_node : output_nodes) {
        GNodePtr cur_node = output_node.first;
        AscendString cur_node_name;
        cur_node->GetName(cur_node_name);
        std::vector<GNodePtr> middle_nodes;
        bool find_matmul_node = true;
        for (int32_t depth = 0; depth < middle_nodes_count; depth++) {
            GNodePtr next_node;
            auto ret = FindSingleDownStreamNodes(cur_node, next_node);
            if (ret != GRAPH_SUCCESS) {
                find_matmul_node = false;
                std::cout << "find single down stream nodes failed" << std::endl;
                break;
            }
            AscendString next_node_type;
            next_node->GetType(next_node_type);
            if (next_node_type != kTypeReshape && next_node_type != kTypeSqueeze && depth != middle_nodes_count - 1) {
                std::cout << "dtype check failed" << std::endl;
                find_matmul_node = false;
                break;
            }

            middle_nodes.emplace_back(cur_node);
            cur_node = next_node;
        }

        if (!find_matmul_node) {
            continue;
        }

        AscendString cur_node_type;
        cur_node->GetType(cur_node_type);
        if (cur_node_type != kTypeMatmulV2 && cur_node_type != kTypeMatmulV3) {
            continue;
        }

        cur_node->GetName(cur_node_name);
        if (cur_node_name == context.base_node_name) {
            continue;
        }

        auto mm_input0_node = cur_node->GetInDataNodesAndPortIndexs(0).first;
        auto mm_input1_node = cur_node->GetInDataNodesAndPortIndexs(1).first;

        AscendString mm_input0_type;
        AscendString mm_input1_type;
        mm_input0_node->GetType(mm_input0_type);
        mm_input1_node->GetType(mm_input1_type);

        int32_t const_idx = -1;
        if (CheckConstType(mm_input0_type)) {
            const_idx = 0;
        } else if (CheckConstType(mm_input1_type)) {
            const_idx = 1;
        } else {
            continue;
        }

        if (const_idx != context.const_idx) {
            continue;
        }

        bool transpose_x1;
        bool transpose_x2;
        if (cur_node->GetAttr(AscendString("transpose_x1"), transpose_x1) != GRAPH_SUCCESS) {
            std::cout << "cur node get transpose_x1 attr failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (cur_node->GetAttr(AscendString("transpose_x2"), transpose_x2) != GRAPH_SUCCESS) {
            std::cout << "cur node get transpose_x2 attr failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (transpose_x1 != context.transpose_x1 || transpose_x2 != context.transpose_x2) {
            continue;
        }

        TensorDesc data_input_desc;
        if (cur_node->GetInputDesc(context.data_idx, data_input_desc) != GRAPH_SUCCESS) {
            std::cout << "get candidate inputdesc failed" << std::endl;
            return GRAPH_FAILED;
        }
        Shape data_input_shape = data_input_desc.GetShape();
        if (!CompareShape(data_input_shape, context.cur_data_shape)) {
            continue;
        }

        int32_t combined_dim;
        if (GetCombinedDim(cur_node, context.data_idx, combined_dim) != GRAPH_SUCCESS) {
            std::cout << "get candidata combined dim failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (combined_dim != context.combined_dim) {
            continue;
        }

        context.AddNode(cur_node, context.data_idx);
        context.AddMiddleNode(middle_nodes);
    }
    return GRAPH_SUCCESS;
}

graphStatus AddReshapeNode(GraphPtr& graph, MmoeMatmulPassContext context, GNode& root_node, GNode& new_mm_node)
{
    if (context.middle_tensors.size() == 0) {
        if (graph->AddDataEdge(root_node, context.root_output_idx, new_mm_node, context.data_idx) != GRAPH_SUCCESS) {
            std::cout << "add edge root node to matmul data failed" << std::endl;
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }
    AscendString mm_name;
    if (new_mm_node.GetName(mm_name) != GRAPH_SUCCESS) {
        std::cout << "get new mm node name failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc root_output_desc;
    if (root_node.GetOutputDesc(context.root_output_idx, root_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get root output desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc reshape_data_input_desc(root_output_desc);

    int32_t reshape_out_dims = context.cur_data_shape.GetDimNum();

    std::string reshape_name = std::string(mm_name.GetString()) + "/Reshape";
    op::Reshape reshape_op(reshape_name.c_str());
    GNode reshape_gnode = graph->AddNodeByOp(reshape_op);
    std::string reshape_shape_node_name = reshape_name + "/Shape";
    op::Const reshape_shape_op(reshape_shape_node_name.c_str());
    reshape_op.SetInput(reshapeShapeName, reshape_shape_op);
    TensorDesc reshape_shape_desc(ge::Shape({reshape_out_dims}), FORMAT_ND, DT_INT32);
    reshape_shape_desc.SetOriginShape(ge::Shape({reshape_out_dims}));

    std::vector<int64_t> reshape_shape_int64 = context.cur_data_shape.GetDims();
    std::vector<int32_t> reshape_shape;
    reshape_shape.reserve(reshape_shape_int64.size());
    for (auto dim : reshape_shape_int64) {
        reshape_shape.emplace_back(static_cast<int32_t>(dim));
    }

    Tensor reshape_shape_tensor(reshape_shape_desc, reinterpret_cast<const uint8_t*>(reshape_shape.data()),
                                reshape_shape.size() * sizeof(int32_t));
    reshape_shape_op.set_attr_value(reshape_shape_tensor);
    GNode reshape_shape_gnode = graph->AddNodeByOp(reshape_shape_op);

    if (graph->AddDataEdge(root_node, context.root_output_idx, reshape_gnode, 0) != GRAPH_SUCCESS) {
        std::cout << "add edge root node to reshape failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(reshape_shape_gnode, 0, reshape_gnode, 1) != GRAPH_SUCCESS) {
        std::cout << "add edge reshape shape to reshape failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(reshape_gnode, 0, new_mm_node, context.data_idx) != GRAPH_SUCCESS) {
        std::cout << "add edge reshape to matmul data failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (reshape_shape_op.UpdateOutputDesc(static_cast<uint32_t>(0), reshape_shape_desc) != GRAPH_SUCCESS) {
        std::cout << "update reshape shape op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (reshape_shape_op.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infershape to reshape shape failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (reshape_op.UpdateInputDesc(static_cast<uint32_t>(0), reshape_data_input_desc) != GRAPH_SUCCESS) {
        std::cout << "update reshape x op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (reshape_op.UpdateInputDesc(static_cast<uint32_t>(1), reshape_shape_desc) != GRAPH_SUCCESS) {
        std::cout << "update reshape shape op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (reshape_op.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infershape to reshape failed" << std::endl;
        return GRAPH_FAILED;
    }
    reshape_op.BreakConnect();
    return GRAPH_SUCCESS;
}

graphStatus ReplaceMmNodes(GraphPtr& graph, GNode& mm_node, CustomPassContext& context, int64_t& replaced_count)
{
    AscendString mm_name;
    auto ret = mm_node.GetName(mm_name);
    if (ret != GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }
    std::string new_mm_name = std::string(mm_name.GetString());

    std::shared_ptr<GNode> data_input;
    int32_t peer_idx;
    int32_t data_idx = 0;
    auto mm_input_node0 = mm_node.GetInDataNodesAndPortIndexs(0);
    auto mm_input_node1 = mm_node.GetInDataNodesAndPortIndexs(1);

    AscendString data_input0_type;
    AscendString data_input1_type;
    mm_input_node0.first->GetType(data_input0_type);
    mm_input_node1.first->GetType(data_input1_type);
    if (CheckConstType(data_input0_type) && !CheckConstType(data_input1_type)) {
        data_input = mm_input_node1.first;
        peer_idx = mm_input_node1.second;
        data_idx = 1;
    } else if (CheckConstType(data_input1_type) && !CheckConstType(data_input0_type)) {
        data_input = mm_input_node0.first;
        peer_idx = mm_input_node0.second;
    } else {
        return GRAPH_SUCCESS;
    }

    int32_t const_idx = 1 - data_idx;
    bool transpose_mm;
    bool transpose_const;
    bool transpose_x1;
    bool transpose_x2;
    if (mm_node.GetAttr(AscendString("transpose_x1"), transpose_x1) != GRAPH_SUCCESS) {
        std::cout << "mm node get adj_x1 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (mm_node.GetAttr(AscendString("transpose_x2"), transpose_x2) != GRAPH_SUCCESS) {
        std::cout << "mm node get adj_x2 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (data_idx == 0) {
        transpose_mm = transpose_x1;
        transpose_const = transpose_x2;
    } else {
        transpose_mm = transpose_x2;
        transpose_const = transpose_x1;
    }

    TensorDesc mm_input_desc;
    Shape mm_input_shape;
    if (mm_node.GetInputDesc(data_idx, mm_input_desc) != GRAPH_SUCCESS) {
        std::cout << "mm node get data desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    mm_input_shape = mm_input_desc.GetShape();

    int32_t k_dim;
    if (transpose_mm == (data_idx == 0)) {
        k_dim = mm_input_shape.GetDim(0);
    } else {
        k_dim = mm_input_shape.GetDim(1);
    }

    int32_t concat_dim;
    if (transpose_const == (const_idx == 0)) {
        concat_dim = 1;
    } else {
        concat_dim = 0;
    }

    int32_t combined_dim;
    if (GetCombinedDim(std::make_shared<GNode>(mm_node), data_idx, combined_dim) != GRAPH_SUCCESS) {
        std::cout << "get mm combined dim failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (combined_dim > k_dim) {
        return GRAPH_SUCCESS;
    }
    GNode root_node;
    std::vector<GNodePtr> middle_nodes;
    int32_t root_output_idx = peer_idx;
    if (FindRootNode(*data_input, root_node, middle_nodes, root_output_idx) != GRAPH_SUCCESS) {
        std::cout << "find root node failed" << std::endl;
        return GRAPH_FAILED;
    }
    int32_t middle_nodes_count = middle_nodes.size();

    AscendString root_node_name;
    root_node.GetName(root_node_name);
    TensorDesc root_output_desc;
    if (root_node.GetOutputDesc(root_output_idx, root_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get root node output desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    Shape root_output_shape = root_output_desc.GetShape();
    MmoeMatmulPassContext fusion_context(root_output_shape, root_output_idx, data_idx, const_idx, transpose_x1,
                                         transpose_x2, combined_dim, mm_input_shape, mm_name);
    fusion_context.AddNode(std::make_shared<GNode>(mm_node), data_idx);
    fusion_context.AddMiddleNode(middle_nodes);

    if (FindMatmulNodes(root_node, middle_nodes_count, fusion_context) != GRAPH_SUCCESS) {
        std::cout << "find matmul nodes failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (fusion_context.matmul_node_context.size() <= 1) {
        return GRAPH_SUCCESS;
    }

    DataType mm_input_dtype = mm_input_desc.GetDataType();
    uint8_t* combined_data;

    if (mm_input_dtype != DT_FLOAT) {
        std::cout << "input dtpye is not float32" << std::endl;
        return GRAPH_SUCCESS;
    }

    int32_t combined_dims = fusion_context.combined_dim * fusion_context.matmul_node_context.size();
    if (concat_matrics<float>(fusion_context, k_dim, concat_dim, combined_data) != GRAPH_SUCCESS) {
        std::cout << "concat_matrics failed" << std::endl;
        return GRAPH_FAILED;
    }

    Shape new_weight_shape;
    if ((data_idx == 0) == !transpose_const) {
        new_weight_shape = Shape({k_dim, combined_dims});
    } else {
        new_weight_shape = Shape({combined_dims, k_dim});
    }
    std::string new_weight_name = std::string(mm_name.GetString()) + "/new_weight";
    op::Const new_weight(new_weight_name.c_str());
    TensorDesc new_weight_desc(new_weight_shape, FORMAT_ND, DT_FLOAT);
    new_weight_desc.SetOriginShape(new_weight_shape);
    Tensor new_weight_tensor(new_weight_desc, combined_data, k_dim * combined_dims * sizeof(float));
    delete[] combined_data;
    new_weight.set_attr_value(new_weight_tensor);
    GNode new_weight_gnode = graph->AddNodeByOp(new_weight);
    TensorDesc new_weight_gnode_desc(new_weight_desc);
    if (new_weight_gnode.UpdateOutputDesc(0, new_weight_gnode_desc) != GRAPH_SUCCESS) {
        std::cout << "update new weight output op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    std::string new_matmul_name = std::string(mm_name.GetString()) + "_combined";
    op::MatMulV2 new_matmul_op(new_matmul_name.c_str());
    GNode new_matmul_gnode = graph->AddNodeByOp(new_matmul_op);
    if (graph->AddDataEdge(new_weight_gnode, 0, new_matmul_gnode, const_idx) != GRAPH_SUCCESS) {
        std::cout << "add edge new const to new matmul failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc new_matmul_data_desc(mm_input_desc);
    TensorDesc new_matmul_const_desc(new_weight_desc);
    if (new_matmul_gnode.UpdateInputDesc(data_idx, new_matmul_data_desc) != GRAPH_SUCCESS) {
        std::cout << "update new matmul data input desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (new_matmul_gnode.UpdateInputDesc(const_idx, new_matmul_const_desc) != GRAPH_SUCCESS) {
        std::cout << "update new matmul const desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (new_matmul_gnode.SetAttr(AscendString("transpose_x1"), transpose_x1) != GRAPH_SUCCESS) {
        std::cout << "mm node set adj_x1 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (new_matmul_gnode.SetAttr(AscendString("transpose_x2"), transpose_x2) != GRAPH_SUCCESS) {
        std::cout << "mm node set adj_x2 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (new_matmul_op.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer new matmul op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (AddReshapeNode(graph, fusion_context, root_node, new_matmul_gnode) != GRAPH_SUCCESS) {
        std::cout << "add reshape node failed" << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc new_matmul_output_desc;
    if (new_matmul_gnode.GetOutputDesc(0, new_matmul_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get matmul output desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    std::string split_name = std::string(mm_name.GetString()) + "/split";
    std::string split_dim_name = split_name + "/split_dim";
    op::Const split_dim_op(split_dim_name.c_str());
    TensorDesc split_dim_desc(ge::Shape({1}), FORMAT_ND, DT_INT32);
    split_dim_desc.SetOriginShape(ge::Shape({1}));
    int32_t split_dims = const_idx;
    Tensor split_dim_tensor(split_dim_desc, reinterpret_cast<const uint8_t*>(&split_dims), sizeof(int32_t));
    split_dim_op.set_attr_value(split_dim_tensor);
    GNode split_dim_gnode = graph->AddNodeByOp(split_dim_op);
    TensorDesc split_dim_gnode_desc(split_dim_desc);
    if (split_dim_gnode.UpdateOutputDesc(0, split_dim_gnode_desc) != GRAPH_SUCCESS) {
        std::cout << "update split dim output op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    std::string size_splits_name = split_name + "/size_splits";
    op::Const size_splits_op(size_splits_name.c_str());
    TensorDesc size_splits_desc(ge::Shape({fusion_context.matmul_node_context.size()}), FORMAT_ND, DT_INT32);
    size_splits_desc.SetOriginShape(ge::Shape({fusion_context.matmul_node_context.size()}));
    std::vector<int32_t> size_splits;
    for (auto rel : fusion_context.matmul_node_context) {
        size_splits.emplace_back(fusion_context.combined_dim);
    }
    Tensor size_splits_tensor(size_splits_desc, reinterpret_cast<const uint8_t*>(size_splits.data()),
                              size_splits.size() * sizeof(int32_t));
    size_splits_op.set_attr_value(size_splits_tensor);
    GNode size_splits_gnode = graph->AddNodeByOp(size_splits_op);
    TensorDesc size_splits_gnode_desc(size_splits_desc);
    if (size_splits_gnode.UpdateOutputDesc(0, size_splits_gnode_desc) != GRAPH_SUCCESS) {
        std::cout << "update size splits output op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    op::SplitV split_op(split_name);
    split_op.SetInput("split_dim", split_dim_op);
    split_op.SetInput("size_splits", size_splits_op);
    split_op.set_attr_num_split(fusion_context.matmul_node_context.size());
    TensorDesc split_node_dim_desc(split_dim_gnode_desc);
    TensorDesc split_node_size_split_desc(size_splits_desc);
    TensorDesc split_node_data_desc(new_matmul_output_desc);
    split_op.DynamicOutputRegister("y", fusion_context.matmul_node_context.size(), false);
    GNode split_gnode = graph->AddNodeByOp(split_op);
    if (graph->AddDataEdge(split_dim_gnode, 0, split_gnode, 2) != GRAPH_SUCCESS) {
        std::cout << "add edge split dim to split gnode failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(size_splits_gnode, 0, split_gnode, 1) != GRAPH_SUCCESS) {
        std::cout << "add size split node to split gnode failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(new_matmul_gnode, 0, split_gnode, 0) != GRAPH_SUCCESS) {
        std::cout << "add edge matmul to split gnode failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (split_gnode.UpdateInputDesc(0, split_node_data_desc) != GRAPH_SUCCESS) {
        std::cout << "update split node data input op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (split_gnode.UpdateInputDesc(1, split_node_size_split_desc) != GRAPH_SUCCESS) {
        std::cout << "update split node size splits input op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (split_gnode.UpdateInputDesc(2, split_node_dim_desc) != GRAPH_SUCCESS) {
        std::cout << "update split node split dim input op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (split_op.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer split op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }
    split_op.BreakConnect();

    for (int32_t i = 0; i < fusion_context.middle_tensors.size(); i++) {
        if (graph->RemoveNode(*fusion_context.middle_tensors[i]) != GRAPH_SUCCESS) {
            std::cout << "remove middle node failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    for (int32_t i = 0; i < fusion_context.matmul_node_context.size(); i++) {
        auto matmul_context = fusion_context.matmul_node_context[i];
        auto ori_const_node = matmul_context.node->GetInDataNodesAndPortIndexs(const_idx).first;
        auto ori_const_outputs = ori_const_node->GetOutDataNodesAndPortIndexs(0);
        if (ori_const_outputs.size() == 1) {
            AscendString const_name;
            ori_const_node->GetName(const_name);
            if (graph->RemoveNode(*ori_const_node) != GRAPH_SUCCESS) {
                std::cout << "remove ori const node failed" << std::endl;
                return GRAPH_FAILED;
            }
        }

        auto mm_outputs = matmul_context.node->GetOutDataNodesAndPortIndexs(0);
        if (graph->RemoveNode(*matmul_context.node) != GRAPH_SUCCESS) {
            std::cout << "remove ori matmul node failed" << std::endl;
            return GRAPH_FAILED;
        }
        for (auto mm_output_node : mm_outputs) {
            if (graph->AddDataEdge(split_gnode, i, *mm_output_node.first, mm_output_node.second) != GRAPH_SUCCESS) {
                std::cout << "add edge split to ori output edge failed" << std::endl;
                return GRAPH_FAILED;
            }
        }
    }

    replaced_count++;

    return GRAPH_SUCCESS;
}

graphStatus MmoeMatmulPass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------MmoeMatmulPass begin----------" << std::endl;
    std::vector<GNode> mm_nodes;
    FindNodesCanFusion(graph, mm_nodes);

    if (mm_nodes.empty()) {
        std::cout << "Not found mm node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;

    for (GNode& node : mm_nodes) {
        if (ReplaceMmNodes(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            std::cout << "replace mm_nodes failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    std::cout << "MMoeMatmulPass end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("MmoeMatmulPass").CustomPassFn(MmoeMatmulPass).Stage(CustomPassStage::kAfterInferShape);
