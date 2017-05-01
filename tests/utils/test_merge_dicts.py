# Copyright 2017, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import
from copy import copy
import unittest

from google.gax.utils import merge_dicts


class MergeDictsTests(unittest.TestCase):
    def test_merge_single_dict(self):
        d = {'foo': 'bar', 'baz': 'bacon', 'spam': 42}
        assert d == merge_dicts(d)

    def test_merge_two_dicts_primitives(self):
        left = {'foo': 'bar', 'baz': 'bacon', 'spam': 42}
        right = {'foo': 3, 'baz': 'bar'}
        assert merge_dicts(left, right) == {
            'baz': 'bar',
            'foo': 3,
            'spam': 42,
        }

    def test_merge_three_dicts_primitives(self):
        left = {'foo': 'bar', 'time': 1335020400}
        middle = {'baz': 'bacon', 'spam': 42}
        right = {'foo': 'blah', 'something': 'else', 'spam': 'eggs'}
        assert merge_dicts(left, middle, right) == {
            'baz': 'bacon',
            'foo': 'blah',
            'something': 'else',
            'spam': 'eggs',
            'time': 1335020400,
        }

    def test_merge_lists(self):
        # Lists act like primitives; the right-hand one wins.
        left = {'foo': ['bar', 'baz'], 'baz': 'bacon'}
        right = {'foo': ['spam', 'eggs']}
        assert merge_dicts(left, right) == {
            'baz': 'bacon',
            'foo': ['spam', 'eggs'],
        }

    def test_merge_nested_dicts(self):
        left = {'foo': {'baz': 'bacon', 'bar': {'spam': 'eggs'}}}
        right = {'foo': {'bar': {'nom': 'nom'}}}
        assert merge_dicts(left, right) == {
            'foo': {
                'bar': {'spam': 'eggs', 'nom': 'nom'},
                'baz': 'bacon',
            },
        }
