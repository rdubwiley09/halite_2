"""
This bot iterates over four features:
    1) open docks on friendly planets (positive)
    2) Unclaimed planets (positive)
    3) friendly ships (negative)
    4) enemy undocked ships (positive)
    5) enemy docked ships (positive)

We use the inverse of the squared distance with a direction to form a vector for each.

From here, the algorithm is as follows:
    1) If able to dock, then dock
    2) Else, follow our vector at max speed avoiding collision

We begin by giving arbitrary weights to each: [3,5,3,1,3]

Eventually, we will determine these weights using reinforcement learning
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import math

from functions.vector import add_vectors, resize_vector

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("Vectorizor")
# Then we print our start message to the logs
logging.info("Starting my Vectorizor bot!")

WEIGHT_CONSTANTS = {
    "friendly_open_docks": 500,
    "unclaimed_planets": 10000,
    "friendly_ships": 100,
    "enemy_ships": 50
}

while True:
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    command_queue = []

    #Initialize our entities of interest
    friendly_open_docks = []
    unclaimed_planets = []
    friendly_ships = []
    enemy_ships = []

    #For every planet
    for planet in game_map.all_planets():
        #The planet is unclaimed
        if not planet.is_owned():
            unclaimed_planets.append(planet)

        #It's one of our planets
        elif planet.owner == game_map.my_id:
            #The planet has open docks
            if not planet.is_full():
                open_docks = planet.num_docking_spots - len(planet.all_docked_ships())
                #Because we just have to be in range of the planet. Go towards the planet.
                for _ in range(open_docks):
                    friendly_open_docks.append(planet)

    for player in game_map.all_players():
        if player.id != game_map.my_id:
            ships = player.all_ships()
            for ship in ships:
                enemy_ships.append(ship)

    friendly_ships = game_map.get_me().all_ships()
    for ship in friendly_ships:
        # Dock them ships brah
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            # Skip this ship
            continue
        #Can't dock so let's move this ship somewhere
        else:
            friendly_open_docks_vector = [0,0]
            unclaimed_planets_vector = [0,0]
            friendly_ships_vector = [0,0]
            enemy_ships_vector = [0,0]

            for friendly_dock in friendly_open_docks:
                distance = ship.calculate_distance_between(friendly_dock)
                angle = ship.calculate_angle_between(friendly_dock)
                if distance != 0:
                    friendly_open_docks_vector = add_vectors(friendly_open_docks_vector,[1/(distance*distance),angle])
            for unclaimed_planet in unclaimed_planets:
                distance = ship.calculate_distance_between(unclaimed_planet)
                angle = ship.calculate_angle_between(unclaimed_planet)
                if distance != 0:
                    unclaimed_planets_vector = add_vectors(unclaimed_planets_vector,[1/(distance*distance),angle])
            for friendly_ship in friendly_ships:
                distance = ship.calculate_distance_between(friendly_ship)
                #Correction to reverse the vector
                angle = (ship.calculate_angle_between(friendly_ship)+180)%360
                if distance != 0:
                    friendly_ships_vector = add_vectors(friendly_ships_vector,[1/(distance*distance),angle])
            for enemy_ship in enemy_ships:
                distance = ship.calculate_distance_between(enemy_ship)
                angle = ship.calculate_angle_between(enemy_ship)
                if distance != 0:
                    enemy_ships_vector = add_vectors(enemy_ships_vector,[1/(distance*distance),angle])

            friendly_open_docks_vector = resize_vector(friendly_open_docks_vector,WEIGHT_CONSTANTS["friendly_open_docks"])
            unclaimed_planets_vector = resize_vector(unclaimed_planets_vector,WEIGHT_CONSTANTS["unclaimed_planets"])
            friendly_ships_vector = resize_vector(friendly_ships_vector,WEIGHT_CONSTANTS["friendly_ships"])
            enemy_ships_vector = resize_vector(enemy_ships_vector,WEIGHT_CONSTANTS["enemy_ships"])

            #TODO: make generic vector addition function
            output_vector = [0,0]
            output_vector = add_vectors(output_vector,friendly_open_docks_vector)
            output_vector = add_vectors(output_vector,unclaimed_planets_vector)
            output_vector = add_vectors(output_vector,friendly_ships_vector)
            output_vector = add_vectors(output_vector,enemy_ships_vector)

            try:
                navigate_command = ship.thrust(
                    magnitude = hlt.constants.MAX_SPEED,
                    angle = output_vector[1]
                )
            except Exception as e:
                navigate_command = ship.thrust(
                    magnitude = hlt.constants.MAX_SPEED,
                    angle = 0
                )
            check = True
            for planet in game_map.all_planets():
                if check:
                    if ship.can_dock(planet):
                        # Need to check if it has any spots
                        if (planet.num_docking_spots - len(planet.all_docked_ships()))/planet.num_docking_spots > .2:
                            if planet.owner == game_map.my_id or not planet.is_owned():
                                navigate_command = ship.dock(planet)
                                check = False
                        #Runs check so the ships don't kamikazee into own planet
                        else:
                            if planet.owner == game_map.my_id:
                                    navigate_command = ship.thrust(
                                        magnitude = 5,
                                        angle = ship.calculate_angle_between(ship.closest_point_to(planet))+90
                                    )
                                    check = False
                                    logging.info("Sending ship away from there")
                    else:
                        distance = ship.calculate_distance_between(ship.closest_point_to(planet))
                        if  distance < hlt.constants.MAX_SPEED:
                            planet_angle = ship.calculate_angle_between(planet)
                            if math.fabs(output_vector[1]-planet_angle)<30:
                                navigate_command = ship.navigate(
                                    ship.closest_point_to(planet),
                                    game_map,
                                    speed=3,
                                    ignore_ships=True
                                )
                                check = False
                                logging.info("Found correct planet")
                            else:
                                navigate_command = ship.thrust(
                                    magnitude = 3,
                                    angle = (planet_angle+output_vector[1])%360/2
                                )
                                check = False
                                logging.info("Slowing ship down")

            command_queue.append(navigate_command)

    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END
# GAME END
