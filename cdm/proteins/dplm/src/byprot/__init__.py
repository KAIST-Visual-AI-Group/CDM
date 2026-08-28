# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


try:
    import byprot.datamodules
except ModuleNotFoundError:
    # The datamodules pull in training-data deps (openfold CUDA kernels) that DPLM-2
    # inference does not need. Skip them when they are unavailable.
    pass
import byprot.models
import byprot.tasks
import byprot.utils
