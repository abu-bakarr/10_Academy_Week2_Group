import imp
import unittest
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, "../scripts/")
sys.path.append(os.path.abspath(os.path.join("scripts")))
from preprocess import Preproccessing


df = pd.DataFrame(
    {
        "decimals": [0.2323, -0.23123, np.NaN, np.NaN, 4.3434],
        "numbers": [2, 4, 6, 7, 9],
        "characters": ["a", "b", "c", "d", "e"],
    }
)


class TestCases(unittest.TestCase):
    def test_dataframe_info(self):
        data_preprocessing = Preproccessing(df)
        self.assertEqual(df.info(), data_preprocessing.df.info())


if __name__ == "__main__":
    unittest.main()
