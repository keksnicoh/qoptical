# -*- coding: utf-8 -*-
""" test the utility module
    :author: keksnicoh
"""

import numpy as np
import pytest
from qoptical.util import *

@pytest.mark.parametrize("a, square", [
    [0,   True],
    [1,   True],
    [1.0, True],
    [4.0, True],
    [4,   True],
    [9,   True],
    [9.2, False],
])
def test_is_square(a, square):
    assert is_square(a) is square

@pytest.mark.parametrize("a, square", [
    [[1, 2, 3, 4],   False],
    [np.array([1, 2, 3, 4]), False],
    [np.array([1, 2, 3, 4]).reshape((2,2)), True],
    [np.array([1, 2, 3, 4, 5, 6]).reshape((3,2)), False],
    [np.matrix([[1,2],[3,4]]), True],
])
def test_is_sqmat(a, square):
    assert is_sqmat(a) is square


@pytest.mark.parametrize("a, b", [
    [1, None],
    [[1,2,3,4], np.matrix([1,2,3,4]).reshape((2,2))],
    [[[1,2],[3,4]], np.matrix([1,2,3,4]).reshape((2,2))],
    [np.array([1,2,3,4]), np.matrix([1,2,3,4]).reshape((2,2))],
    [np.matrix([1,2,3,4]), np.matrix([1,2,3,4]).reshape((2,2))],
    [np.matrix([1,2,3,4]).reshape((1,4)), np.matrix([1,2,3,4]).reshape((2,2))],
    [np.matrix([1,2,3,4]).reshape((1,4)), np.matrix([1,2,3,4]).reshape((2,2))],
    [np.array([1,2,3,4]).reshape((1,2,2)), np.matrix([1,2,3,4]).reshape((2,2))],
    [np.matrix([1,2,3,4,1,2,3,4]).reshape((2,4)), None],
    [np.matrix([1,2,3,4,5,6]).reshape((2,3)), None],
    [np.matrix([1,2,3,4,5,6]).reshape((2,1,1,3)), None],
    [np.array([1,2,3,4,5,6,7,8]).reshape((2,2,2)), None],
])
def test_sqmat(a, b):
    if b is None:
        try:
            sqmat(a)
        except:
            return
        assert False, 'should have failed...'

    c = sqmat(a)
    assert isinstance(c, np.matrix)
    assert c.shape == b.shape
    assert np.all(c == b)

@pytest.mark.parametrize("a, b", [
    [1, np.array([1])],
    [[1], np.array([1])],
    [[1,2], np.array([1,2])],
    [np.array([1,2]), np.array([1,2])],
    [np.array([1,2,3,4]).reshape((2,2)), None],
])
def test_vecorize(a, b):
    if b is None:
        try:
            vectorize(a)
        except:
            return
        assert False, 'should have failed...'

    c = vectorize(a)
    assert isinstance(c, np.ndarray)
    assert c.shape == b.shape
    assert np.all(c == b)

@pytest.mark.parametrize("a, b", [
    [1, None],
    [{}, None],
    [[1], 1],
    [[1, 2], np.array([1,2])],
    [np.array([1]), 1],
    [np.array([1,2]), np.array([1,2])],
])
def test_unvectorize(a, b):
    if b is None:
        try:
            unvectorize(a)
        except:
            return
        assert False, 'should have failed...'

    c = unvectorize(a)
    if isinstance(b, np.ndarray):
        assert isinstance(c, np.ndarray)
        assert c.shape == b.shape
        assert np.all(c == b)
    else:
        assert b == c



