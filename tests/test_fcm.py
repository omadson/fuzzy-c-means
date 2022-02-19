# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCM

# some test data
X = np.random.normal(size=(10, 2))


class TestFCM:
    def test_u_creation(self):
        fcm = FCM()
        fcm.fit(X)
        assert fcm.u is not None

    def test_dont_fit(self):
        fcm = FCM()
        assert fcm.is_trained() == False
        with pytest.raises(ReferenceError):
            partition_entropy_coefficient = fcm.partition_entropy_coefficient
        with pytest.raises(ReferenceError):
            partition_coefficient = fcm.partition_coefficient
        with pytest.raises(ReferenceError):
            centers = fcm.centers

    def test_fit(self):
        fcm = FCM()
        fcm.fit(X)
        assert fcm.u is not None
