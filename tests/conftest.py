import os

os.environ['AEIC_DATA_DIR'] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data')
)

print(os.environ['AEIC_DATA_DIR'])
