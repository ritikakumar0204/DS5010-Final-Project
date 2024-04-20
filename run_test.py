import unittest

# import test modules

from whowrotethis.tests import TestTextEmbeddings, TestTextPreprocessing

# initialize test suite

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite

suite.addTest(loader.loadTestsFromModule(TestTextEmbeddings))
suite.addTest(loader.loadTestsFromModule(TestTextPreprocessing))

# initialize a test runner and run the test suite

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
