import math
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import pyproj
from math import sin, cos, asin, atan2, modf, pi, sqrt
import json

class Assigner:
    def __init__(self, users, satellites, interferers):
        self.users = users
        self.satellites = satellites
        self.interferers = interferers

    def process(self, case):
        print('# You have {u} users, {s} satellites, and {i} interferers.'.format(
            u = len(self.users),
            s = len(self.satellites),
            i = len(self.interferers)
        ))
        self.users.sort(key=lambda x: x.position)
        self.assign(case)

    def assign(self, case):
        assigned_users = []
        assigned_satellites = []

        # populate the users, satellites, and interferers into np arrays
        users = np.array([[u.position.x, u.position.y, u.position.z] for u in self.users])
        satellites = np.array([[s.position.x, s.position.y, s.position.z] for s in self.satellites])
        interferers = np.array([[i.position.x, i.position.y, i.position.z] for i in self.interferers])

        # build the kd trees
        user_tree = cKDTree(users)
        satellite_tree = cKDTree(satellites)

        # query for max distance of 1000m
        indexes = user_tree.query_ball_tree(satellite_tree, r = 1000)

        # loop and attempt to assign
        for i in range(len(indexes)):
            user = self.users[i] # get the user from the index
            s = [self.satellites[s] for s in indexes[i]]
            s.sort(key=lambda x: user.degrees_from(x), reverse = True)
            for satellite in s:
                success = satellite.assign(user, self.interferers) # checks interference with non-starlink
                                                                   # satellites and also checks to see if we
                                                                   # interfere with any other customer on this
                                                                   # satellite (color bands)
                
                # good to go! we have something
                if success:
                    color = success[1]
                    success = success[0]
                    assigned_users.append(user.id)
                    if not satellite.id in assigned_satellites:
                        assigned_satellites.append(satellite.id)
                    print(success)
                    break # we found a satellite, lets move on

        with open('earth.json') as f:
            data = json.load(f)
            d = data.get('data', [])
            l = data.get('layout', {})
            fig = go.Figure(data=d, layout=l)

        # plot the users
        fig.add_scatter3d(
            name = 'users',
            x = users[:, 0],
            y = users[:, 1],
            z = users[:, 2],
            hoverinfo = 'text',
            text = ['user {}<br>{}'.format(x.id, '<br>'.join(x.reasons) if not x.responses else x.responses) for x in self.users],
            mode = 'markers',
            marker = dict(
                symbol = ['circle' if x.responses else 'circle-open' for x in self.users],
                size = 3.2,
                color = 'green'
            )
        )
            
        # plot the satellites
        fig.add_scatter3d(
            name = 'satellites',
            x = satellites[:, 0],
            y = satellites[:, 1],
            z = satellites[:, 2],
            hoverinfo = 'text',
            text = ['satellite {} assigned {}'.format(x.id, x.users()) for x in self.satellites],
            mode = 'markers',
            marker = dict(
                symbol = ['diamond' if x.users() == 32 else 'diamond-open' for x in self.satellites],
                size = 5,
                color = 'red'
            )
        )

        # plot the interferers
        #fig.add_scatter3d(
        #    name = 'interferers',
        #    x = interferers[:, 0],
        #    y = interferers[:, 1],
        #    z = interferers[:, 2],
        #    hoverinfo = 'text',
        #    text = ['interferer {}'.format(x.id) for x in self.interferers],
        #    mode = 'markers',
        #    marker = dict(
        #        symbol = 'diamond-open',
        #        size = 5,
        #        color = 'blue'
        #    )
        #)

        # plot the lines
        for satellite in self.satellites:
            if satellite.users() == 0:
                continue

            # fill the x, y, and z with the users on beams
            x, y, z, colors, texts = [], [], [], [], []
            for beam, users in satellite.beams.items():
                for user in users:
                    x.append(user.position.x)
                    x.append(satellite.position.x)
                    y.append(user.position.y)
                    y.append(satellite.position.y)
                    z.append(user.position.z)
                    z.append(satellite.position.z)
                    colors.append(band_to_color(beam))
                    texts.append(user.responses)

            # add the scatter plot
            fig.add_scatter3d(
                name = 'satellite {} beams'.format(satellite.id),
                x = x,
                y = y,
                z = z,
                mode = 'lines',
                text = texts,
                line = dict(width = 2, color = colors),
                opacity = 0.5
            )

        # update the layout
        fig.update_layout(margin=dict(t=30, r=0, l=20, b=10))

        # write the html
        fig.write_html(case + '.html', auto_open = True)

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

def band_to_color(band):
    if band == 'A':
        return 'white'
    elif band == 'B':
        return 'red'
    elif band == 'C':
        return 'green'
    elif band == 'D':
        return 'blue'