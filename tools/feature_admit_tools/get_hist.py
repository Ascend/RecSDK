import json

import numpy as np

file_name = "slice_0.data"
data = np.fromfile(file_name, dtype=np.int64)
data = data[1:].reshape(-1, 3)
result = {}

with open("admit_hist.json", "w") as f:
    for d in data:
        key, count, _ = d
        result[str(key)] = int(count)

    sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    json.dump(sorted_result, f, indent=4)
