"""
Instead of trying to construct a vector, we use Gaussians to create gradients that push our ships towards good stuff and away from bad stuff.

This bot uses five gaussians:
    1) Dock unowned planet (Gu)
    2) Dock friendly planet (Gf)
    3) Enemy planet (Ge)
    4) Enemy Ship (Ges)
    5) Friendly ship (Gfs)

We then contstruct a gradient function by summing the gaussians centered at each feature.

From here, the algorithm is as follows:
    1) If able to dock, then dock
    2) Else, follow our gradient at max speed avoiding collision

For each gaussian we give it a width parameter.

Eventually, we will determine these parameters using reinforcement learning, but for now, I will use simple weights [1,1,1,1,1]
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import math
from itertools import repeat

from functions.gradient import compute_gradient
from functions.vector import add_vectors

class Bot:
    def __init__(self, weight_parameters):
        """
        self.weight_parameters = {
            "unowned_planet": 1e25,
            "friendly_planet": 1e5,
            "enemy_planet": 1e5,
            "enemy_ship": 1e5,
            "friendly_ship": 1e5
        }
        """
        self.weight_parameters = weight_parameters

    def play(self):

        # GAME START
        # Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
        game = hlt.Game("Gradient Bot")
        # Then we print our start message to the logs
        logging.info("Starting my bot!")

        while True:
            # TURN START
            # Update the map for the new turn and get the latest version
            game_map = game.update_map()

            # Here we define the set of commands to be sent to the Halite engine at the end of the turn
            command_queue = []

            #Initialize our entities of interest
            unowned_planets = []
            friendly_planets = []
            enemy_planets = []
            enemy_ships = []
            friendly_ships = []

            #Get all the reqluired objects
            logging.info("My player id is {}".format(game_map.my_id))
            #For every planet
            for planet in game_map.all_planets():
                #The planet is unclaimed
                logging.info("Checking planet owned by {}".format(planet.owner))
                if not planet.is_owned():
                    logging.info("unowned planet {} has {} docking spots".format(planet.id,planet.num_docking_spots))
                    unowned_planets.extend(repeat(planet,(planet.num_docking_spots)))
                #It's one of our planets
                elif planet.owner.id == game_map.my_id:
                    logging.info("friendly planet {} has {} docking spots and {} current docked ships".format(planet.id,planet.num_docking_spots,len(planet.all_docked_ships())))
                    friendly_planets.extend(repeat(planet,(planet.num_docking_spots-len(planet.all_docked_ships()))))
                else:
                    enemy_planets.extend(repeat(planet,(planet.num_docking_spots)))
            logging.info("{} unowned docks and {} open friendly docks".format(len(unowned_planets),len(friendly_planets)))
            #For all players
            for player in game_map.all_players():
                #Other player
                logging.info(game_map.my_id)
                if player.id != game_map.my_id:
                    #Add these ships to the list
                    enemy_ships += player.all_ships()
                #It's me
                else:
                    #Get my ships
                    friendly_ships += player.all_ships()


            for ship in friendly_ships:
                # Dock them ships brah
                if ship.docking_status != ship.DockingStatus.UNDOCKED:
                    # Skip this ship
                    continue
                #Can't dock so let's move this ship somewhere
                else:
                    vector_x = 0
                    vector_y = 0
                    navigate_command = None
                    for planet in friendly_planets+unowned_planets:
                        if navigate_command:
                            break
                        if ship.can_dock(planet):
                            if not planet.is_full():
                                if not planet.is_owned() or planet.owner.id == game_map.my_id:
                                    navigate_command=ship.dock(planet)
                        else:
                            if not planet.is_full() and ship.calculate_distance_between(ship.closest_point_to(planet)) < hlt.constants.DOCK_RADIUS:
                                closest_point = ship.closest_point_to(planet)
                                opposite_point = hlt.entity.Position((planet.x-closest_point.x)*2,(planet.y-closest_point.y)*2)
                                navigate_command = ship.navigate(
                                            ship.closest_point_to(planet),
                                            game_map,
                                            speed=7,
                                            ignore_ships=False
                                        )
                            else:
                                if planet.is_owned():
                                    x_partial, y_partial = compute_gradient(ship.x,ship.y,planet.x,planet.y,1/self.weight_parameters["friendly_planet"],1)
                                else:
                                    x_partial, y_partial = compute_gradient(ship.x,ship.y,planet.x,planet.y,1/self.weight_parameters["unowned_planet"],1)
                                vector_x += x_partial
                                vector_y += y_partial
                #logging.info("gradient after planets {} {}".format(vector_x,vector_y))
                if navigate_command:
                    command_queue.append(navigate_command)
                    continue
                else:
                    for planet in enemy_planets:
                        x_partial, y_partial = compute_gradient(ship.x,ship.y,planet.x,planet.y,1/self.weight_parameters["enemy_planet"],1)
                        vector_x += x_partial
                        vector_y += y_partial

                    for enemy_ship in enemy_ships:
                        x_partial, y_partial = compute_gradient(ship.x,ship.y,enemy_ship.x,enemy_ship.y,1/self.weight_parameters["enemy_ship"],1)
                        vector_x += x_partial
                        vector_y += y_partial
                    for friendly_ship in friendly_ships:
                        x_partial, y_partial = compute_gradient(ship.x,ship.y,friendly_ship.x,friendly_ship.y,1/self.weight_parameters["friendly_ship"],-1)
                        vector_x += x_partial
                        vector_y += y_partial
                    direction_vector_magnitude = math.sqrt(pow(vector_x,2)+pow(vector_y,2))
                    angle_radians = math.atan2(vector_y,vector_x)
                    angle_degrees = (angle_radians*180)/math.pi
                    direction_vector = [direction_vector_magnitude,angle_degrees]
                    logging.info("direction vector is {}".format(direction_vector))
                    end_x = hlt.constants.MAX_SPEED*math.cos(direction_vector[1])+ship.x
                    end_y = hlt.constants.MAX_SPEED*math.sin(direction_vector[1])+ship.y
                    end_position = hlt.entity.Position(end_x,end_y)

                    for planet in friendly_planets+unowned_planets+enemy_planets:
                        if navigate_command:
                            break
                        if hlt.collision.intersect_segment_circle(ship,end_position,planet):
                            navigate_command = ship.navigate(
                                        end_position,
                                        game_map,
                                        speed=hlt.constants.MAX_SPEED,
                                        ignore_ships=False
                                    )
                    if not navigate_command:
                        navigate_command = ship.thrust(
                            magnitude = hlt.constants.MAX_SPEED,
                            angle = direction_vector[1]
                        )
                    command_queue.append(navigate_command)

            # Send our set of commands to the Halite engine for this turn
            game.send_command_queue(command_queue)
            # TURN END
        # GAME END
