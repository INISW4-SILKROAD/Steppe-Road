import os, sys
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if dir not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_multimodal_encoder import custom_mp