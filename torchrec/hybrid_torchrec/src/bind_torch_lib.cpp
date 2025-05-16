

#include <torch/torch.h>

#include <cstdint>

#include "ids_mapper.h"
#include "ids_process/ids_mapper.h"
#include "unique.h"

TORCH_LIBRARY(hybrid, m)
{
    m.class_<hybrid::IdsMapper>("IdsMapper")
        .def(torch::init<int64_t>())
        .def("ids2indices_unique", &hybrid::IdsMapper::UniqueAndLookup)
        .def("ids2indices_unique_out", &hybrid::IdsMapper::UniqueAndLookupOut);
}