from reinforcement_bot.bot import Bot

weight_parameters = {
    "unowned_planet": 1e3*10,
    "friendly_planet": 1e3,
    "enemy_planet": 1e3*10,
    "enemy_ship": 1e3,
    "friendly_ship": 1e3*100
}

Bot(weight_parameters).play()
