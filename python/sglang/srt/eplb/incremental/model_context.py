# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from contextlib import contextmanager
from contextvars import ContextVar

_BUILDING_INCREMENTAL_EPLB_TARGET = ContextVar(
    "building_incremental_eplb_target", default=False
)


@contextmanager
def incremental_eplb_target_model_context(*, is_target: bool = True):
    token = _BUILDING_INCREMENTAL_EPLB_TARGET.set(is_target)
    try:
        yield
    finally:
        _BUILDING_INCREMENTAL_EPLB_TARGET.reset(token)


def is_building_incremental_eplb_target() -> bool:
    return _BUILDING_INCREMENTAL_EPLB_TARGET.get()
