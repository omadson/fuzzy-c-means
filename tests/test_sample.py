import unittest
import numpy as np

class TestMethods(unittest.TestCase):
    def test_fcm():
        # some test data
        X = np.random.normal(size=(10, 2))
        fcm = FCM()
        fcm.fit(X)
        self.assertTrue(fcm.u is not None)
