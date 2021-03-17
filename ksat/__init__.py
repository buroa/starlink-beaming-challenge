from ksat.assigner import Assigner
from ksat.classes import *

# eval the case
def eval(case):
    
    users, satellites, interferers = [], [], []
    with open(case, 'r') as case:
        users, satellites, interferers = parse(case.read())

    if (len(users) > 0 and len(satellites) > 0):
        assigner = Assigner(users, satellites, interferers)
        assigner.process()
    else:
        print('You need to provide atleast a user and a satellite.')

# parse the case
def parse(case):
    users, satellites, interferers = [], [], []

    # read line by line, ignore, and add to class
    for line in case.splitlines():
        if line.startswith('#') or len(line) == 0:
            continue
        
        of, iden, x, y, z = line.split(' ')
        iden, x, y, z = int(iden), float(x), float(y), float(z)
        pos = Position(x, y, z)

        # build the classes
        if of == 'user':
            users.append(User(iden, pos))
        elif of == 'sat':
            satellites.append(Satellite(iden, pos))
        elif of == 'interferer':
            interferers.append(Interferer(iden, pos))

    return users, satellites, interferers