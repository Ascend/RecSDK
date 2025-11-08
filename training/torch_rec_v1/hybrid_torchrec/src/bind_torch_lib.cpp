

#include <torch/torch.h>

#include <cstdint>

#include "ids_mapper.h"
#include "ids_process/ids_mapper.h"
#include "ids_process/bucketize.h"

TORCH_LIBRARY(hybrid, m)
{
    m.class_<hybrid::IdsMapper>("IdsMapper")
        .def(torch::init<int64_t, bool>())
        .def("ids2indices_unique", &hybrid::IdsMapper::UniqueAndLookup)
        .def("ids2indices_unique_out", &hybrid::IdsMapper::UniqueAndLookupOut)
        .def("export_ids_and_indices", &hybrid::IdsMapper::ExportIdsAndIndices)
        .def("load_original_ids", &hybrid::IdsMapper::LoadOriginalIds)
        .def_static("parallel_ids2indices_unique_out", &hybrid::IdsMapper::ParallelUniqueHashOut);
        
    m.def("block_bucketize_sparse_features_cpu", &hybrid::BlockBucketizeSparseFeaturesCpu);
}