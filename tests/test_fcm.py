"""Tests for FCM Class"""
# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCM

# some test data
X = np.random.normal(size=(10, 2))


def test_u_creation():
    """Test if its generate u matrix"""
    fcm = FCM()
    fcm.fit(X)
    assert fcm.u is not None


def test_dont_fit():
    """Test if its returns Exceptions"""
    fcm = FCM()
    assert fcm._is_trained() == False
    with pytest.raises(ReferenceError):
        partition_entropy_coefficient = fcm.partition_entropy_coefficient
    with pytest.raises(ReferenceError):
        partition_coefficient = fcm.partition_coefficient
    with pytest.raises(ReferenceError):
        centers = fcm.centers
