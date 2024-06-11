import os, sys

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if dir not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier.preprocessor import *