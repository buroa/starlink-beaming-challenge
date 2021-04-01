from ksat.evaluate import *

class Satellite:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.beams = {
            'A': [],
            'B': [],
            'C': [],
            'D': []
        }

    def assign(self, user, interferers):
        assignable = self.users() < 32

        if not assignable:
            user.reason('{satellite} is full.'.format(
                satellite = self,
            ))
            return
            
        point_a = C(user.position.x, user.position.y, user.position.z)
        point_b = C(self.position.x, self.position.y, self.position.z)

        degrees = user.degrees_from(self)
        if degrees <= 135:
            user.reason('We are outside of {satellite} field of view by {degrees} degrees.'.format(
                satellite = self,
                degrees = degrees - 90
            ))
            return

        # check inferferers
        for interferer in interferers:
            point_c = C(interferer.position.x, interferer.position.y, interferer.position.z)
            degrees = J(point_a, point_b, point_c)
            if degrees < 20:
                user.reason('{satellite} interferes with {interferer} by {degrees} degrees.'.format(
                    satellite = self,
                    interferer = interferer,
                    degrees = degrees
                ))
                assignable = False
                break

        if not assignable:
            return

        # check beams and colors
        assigned_beam = None
        for beam, customers in self.beams.items():
            valid = True

            for customer in customers:
                point_c = C(customer.position.x, customer.position.y, customer.position.z)
                degrees = J(point_b, point_c, point_a)
                if degrees < 10:
                    user.reason('We would interfere {customer} on {satellite} by {degrees} degrees.'.format(
                        satellite = self,
                        customer = customer,
                        degrees = degrees
                    ))
                    valid = False
                    break
                
            if valid:
                assigned_beam = beam
                break

        if not assigned_beam:
            return

        # assign the user, actually
        self.beams.get(assigned_beam).append(user)

        # build the response
        response = 'sat {sat} beam {beam} user {user} color {color}'.format(
            sat = self.id,
            beam = self.users(),
            user = user.id,
            color = assigned_beam
        )

        # add the response
        user.response(response)

        # return the response
        return response, assigned_beam

    def users(self):
        return sum(map(len, self.beams.values()))

    def __repr__(self):
        return 'Satellite({id}, {pos})'.format(
            id = self.id, pos = self.position)