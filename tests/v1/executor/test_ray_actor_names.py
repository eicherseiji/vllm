# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.executor.ray_utils import build_actor_name


def test_build_actor_name_preserves_non_dp_names():
    assert build_actor_name("test", rank=0, tp_size=1, pp_size=1, pcp_size=1) == (
        "vllm_Worker_test"
    )
    assert build_actor_name("test", rank=3, tp_size=2, pp_size=2, pcp_size=1) == (
        "vllm_Worker_test_TP1_PP1"
    )
    assert build_actor_name("test", rank=5, tp_size=2, pp_size=1, pcp_size=3) == (
        "vllm_Worker_test_TP1_PCP2"
    )


def test_build_actor_name_includes_dp_rank_for_dp_engines():
    names = [
        build_actor_name(
            "test",
            rank=0,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            data_parallel_rank=dp_rank,
            data_parallel_size=16,
        )
        for dp_rank in range(16)
    ]

    assert len(set(names)) == 16
    assert names[0] == "vllm_Worker_test_DP0"
    assert names[-1] == "vllm_Worker_test_DP15"


def test_build_actor_name_requires_dp_rank_for_dp_engines():
    with pytest.raises(ValueError, match="data_parallel_rank is required"):
        build_actor_name(
            "test",
            rank=0,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            data_parallel_size=2,
        )
