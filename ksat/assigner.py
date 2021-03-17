import math

class Assigner:
    def __init__(self, users, satellites, interferers):
        self.users = users
        self.satellites = satellites
        self.interferers = interferers

    def process(self):
        print('# You have {u} users, {s} satellites, and {i} interferers.'.format(
            u = len(self.users),
            s = len(self.satellites),
            i = len(self.interferers)
        ))
        self.users.sort(key=lambda x: x.position)
        self.assign()

    def assign(self):
        assigned_users = []
        assigned_satellites = []

        for user in self.users:
            satellites = user.within_view(self.satellites)
            if not len(satellites) > 0:
                continue
            
            # sort by field of view
            satellites.sort(key=lambda x: x[1])

            # attempt to assign a satellite
            for satellite, degrees, distance in satellites:
                success = satellite.assign(user, self.interferers) # checks interference with non-starlink
                                                                   # satellites and also checks to see if we
                                                                   # interfere with any other customer on this
                                                                   # satellite (color bands)
                
                # good to go! we have something
                if success:
                    assigned_users.append(user.id)
                    if not satellite.id in assigned_satellites:
                        assigned_satellites.append(satellite.id)
                    print(success)
                    break # we found a satellite, lets move on

        # internal statistics
        if len(assigned_users) > 0 and len(assigned_satellites) > 0:
            print('# Statistics:')
            print('# \tUnassigned Users: {users}.'.format(
                users = len(self.users) - len(assigned_users)
            ))
            print('# \tUnassigned Satellites: {satellites}.'.format(
                satellites = len(self.satellites) - len(assigned_satellites)
            ))
            print('# \tUser Success: {us}%, Satellite Success: {ss}%.'.format(
                us = (len(assigned_users) / len(self.users)) * 100,
                ss = (len(assigned_satellites) / len(self.satellites)) * 100
            ))