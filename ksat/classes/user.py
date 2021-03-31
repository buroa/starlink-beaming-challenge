import math
import numpy as np

class User:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.reasons = []
        self.responses = ''

    def reason(self, r):
        self.reasons.append(r)

    def response(self, r):
        self.responses = r

    def degrees_from(self, other):
        user_np = self.position.to_np()
        v1 = other.position.to_np() - user_np
        v2 = -user_np
        angle = math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle * 180.0 / math.pi

    def __eq__(self, other):
        if self.id != other.id:
            return False
        if self.position != other.position:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return ((self.degrees_from(other)) < (other.degrees_from(self)))

    def __repr__(self):
        return 'User({id}, {pos})'.format(
            id = self.id, pos = self.position)