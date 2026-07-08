import unittest

from tests.unittests.io import SPECTRAUFIOUnitTest
from tests.unittests.util import SPECTRAUFUtilUnitTest
from tests.unittests.regularization import SPECTRAUFRegularizationUnitTest
from tests.regression.regression import SPECTRAUFUnitTest5Channel


def main():
    """ unit tests
    
    """
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SPECTRAUFIOUnitTest))
    suite.addTest(unittest.makeSuite(SPECTRAUFUtilUnitTest))
    suite.addTest(unittest.makeSuite(SPECTRAUFUnitTest5Channel))
    suite.addTest(unittest.makeSuite(SPECTRAUFRegularizationUnitTest))

if __name__ == '__main__':
    unittest.main()
