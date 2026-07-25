# coding=utf-8
# Copyright 2026 The Perch Authors.
#
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

"""SQLite database implementation using USearch for vector storage & search."""

import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import functools
import itertools
import json
import re
import sqlite3
import threading
from typing import Any, Literal

from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface
from perch_hoplite.db import score_functions
from perch_hoplite.db import search_results
from usearch import index as uindex