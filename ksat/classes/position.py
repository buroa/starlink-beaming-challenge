import numpy as np

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, pos):
        return Position(abs(pos.x - self.x), abs(pos.y - self.y), abs(pos.z - self.z))

    def __eq__(self, other):
        return ((self.x, self.y, self.z) == (other.x, other.y, other.z))

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return ((self.x, self.y, self.z) < (other.x, other.y, other.z))

    def to_np(self):
        return np.array((self.x, self.y, self.z))

    def __repr__(self):
        return 'Position({x}, {y}, {z})'.format(
            x = self.x, y = self.y, z = self.z) 