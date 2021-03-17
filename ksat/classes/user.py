import math
import numpy as np

class User:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.reasons = []

    def reason(self, r):
        self.reasons.append(r)

    def degrees_from(self, satellite):
        user_np = self.position.to_np()
        v1 = satellite.position.to_np() - user_np
        v2 = -user_np
        angle = math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle * 180.0 / math.pi

    def within_view(self, satellites):
        response = []
        for i, satellite in enumerate(satellites):
            distance = satellite.position.distance(self.position)
            if distance.x >= 1000 or distance.y >= 1000 or distance.z >= 1000: 
                continue
            degrees = self.degrees_from(satellite)
            if degrees > 135:
                response.append((satellite, degrees, distance))
        return response

    def __eq__(self, other):
        if self.id != other.id:
            return False
        if self.position != other.position:
            return False
        return True

    def __repr__(self):
        return 'User({id}, {pos})'.format(
            id = self.id, pos = self.position)