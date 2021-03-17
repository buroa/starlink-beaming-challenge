class Interferer:
    def __init__(self, id, position):
        self.id = id
        self.position = position

    def __repr__(self):
        return 'Interferer({id}, {pos})'.format(
            id = self.id, pos = self.position)