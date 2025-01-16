from pickle import FALSE
from kesslergame import KesslerController
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

class WhiteDoveController(KesslerController):
    def __init__(self, chromosome = None):
        # from using easy ga, this chromosome hit the most asteroids and survived the longest
        # asteroids hit: 491
        if chromosome is None:
          self.chromosome = [7, 100, 153, 0, 162, 181]
        else:
          self.chromosome = chromosome.gene_value_list[0]

        self.eval_frames = 0 #What is this?
        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.01), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python

        asteroid_distance = ctrl.Antecedent(np.arange(0, 700, 50), 'asteroid_distance')
        ship_speed = ctrl.Antecedent(np.arange(-240, 240, 10), 'ship_speed')
        ship_thrust = ctrl.Consequent(np.arange(-480, 480, 20), 'ship_thrust')

        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1.0,1.0,0.1), 'ship_fire')

        ship_speed = ctrl.Antecedent(np.arange(0.0, 240.0, 5), 'ship_speed')
        mine_approach = ctrl.Antecedent(np.arange(0, 16, 1), 'mine_approach')
        mine_fire = ctrl.Consequent(np.arange(-1.0,1.0,0.1), 'mine_fire')

        mine_fire['N'] = fuzz.trimf(mine_fire.universe, [-1,-1,0.0])
        mine_fire['Y'] = fuzz.trapmf(mine_fire.universe, [-0.1, 0.0, 1.0, 1.0])

        #Declare fuzzy sets for impact_approach (total asteroids in 150 m radius)
        mine_approach['L'] = fuzz.trimf(mine_approach.universe, [5, 10, 10]) #Lot
        mine_approach['S'] = fuzz.trimf(mine_approach.universe, [0, 5, 10]) #Some
        mine_approach['F'] = fuzz.trimf(mine_approach.universe, [0, 0, 5]) #NO IMPACT


        #Declare fuzzy sets for mine mine_speed (velocity of the ship)
        ship_speed['SM'] = fuzz.trapmf(ship_speed.universe, [0.0, 0.0, 80.0, 100.0]) #STATIC MOVEMENT
        ship_speed['CM'] = fuzz.trimf(ship_speed.universe, [0.0, 100.0, 240.0]) #CALCULATED MOVEMENT
        ship_speed['EM'] = fuzz.trimf(ship_speed.universe, [100.0, 240.0, 240.0]) #EVASIVE MOVEMENT

        distance_max_threshold = 800
        distance_large_threshold = 400
        distance_mid_threshold = 200
        distance_small_threshold = 100
        asteroid_distance['VC'] = fuzz.trimf(asteroid_distance.universe, [0, 0, distance_small_threshold])
        asteroid_distance['C'] = fuzz.trimf(asteroid_distance.universe, [0, distance_small_threshold, distance_mid_threshold])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe,[distance_small_threshold, distance_mid_threshold, distance_large_threshold])
        asteroid_distance['F'] = fuzz.trimf(asteroid_distance.universe,[distance_mid_threshold, distance_large_threshold, distance_max_threshold])
        asteroid_distance['VF'] = fuzz.trimf(asteroid_distance.universe,[distance_large_threshold, distance_max_threshold, distance_max_threshold])

        speed_max = 200
        speed_low = self.chromosome[0]
        speed_mid = self.chromosome[1]
        speed_high = self.chromosome[2]
        ship_speed['NL'] = fuzz.trimf(ship_speed.universe,[-speed_max, -speed_max, -speed_mid])
        ship_speed['NS'] = fuzz.trimf(ship_speed.universe, [-speed_high, -speed_mid, -speed_low])
        ship_speed['Z'] = fuzz.trimf(ship_speed.universe, [-speed_mid, speed_low, speed_mid])
        ship_speed['PS'] = fuzz.trimf(ship_speed.universe, [speed_low, speed_mid, speed_high])
        ship_speed['PL'] = fuzz.trimf(ship_speed.universe,[speed_mid, speed_max, speed_max])

        thrust_max_point = 250
        thrust_low_point = self.chromosome[3]
        thrust_mid_point = self.chromosome[4]
        thrust_high_point = self.chromosome[5]
        ship_thrust['NL'] = fuzz.trimf(ship_thrust.universe, [-thrust_max_point, -thrust_max_point, -thrust_mid_point])
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [-thrust_high_point, -thrust_mid_point, thrust_low_point])
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [-thrust_mid_point, thrust_low_point, thrust_mid_point])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [thrust_low_point, thrust_mid_point, thrust_high_point])
        ship_thrust['PL'] = fuzz.trimf(ship_thrust.universe, [thrust_mid_point, thrust_max_point, thrust_max_point])

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be thresholded
        # and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        # rules added for thrust controller which takes the asteroid distance, ship speed, and theta_delta as inputs
        # first part takes the asteroid distance and the orientation of the ship in realtion to the closest asteroid
        # second part takes the ship speed and the distance of the closest asteroid to determine how much thrust to apply
        self.thrusting_control = ctrl.ControlSystem([
            ctrl.Rule(asteroid_distance['M'] & theta_delta['NL'], ship_thrust['PL']),
            ctrl.Rule(asteroid_distance['M'] & theta_delta['NS'], ship_thrust['PS']),
            ctrl.Rule(asteroid_distance['M'] & theta_delta['Z'], ship_thrust['PS']),
            ctrl.Rule(asteroid_distance['M'] & theta_delta['PS'], ship_thrust['PS']),
            ctrl.Rule(asteroid_distance['M'] & theta_delta['PL'], ship_thrust['PL']),
            ctrl.Rule((asteroid_distance['C'] | asteroid_distance['VC']) & theta_delta['NL'], ship_thrust['PS']),
            ctrl.Rule((asteroid_distance['C'] | asteroid_distance['VC']) & theta_delta['NS'], ship_thrust['NS']),
            ctrl.Rule((asteroid_distance['C'] | asteroid_distance['VC']) & theta_delta['Z'], ship_thrust['NL']),
            ctrl.Rule((asteroid_distance['C'] | asteroid_distance['VC']) & theta_delta['PS'], ship_thrust['NS']),
            ctrl.Rule((asteroid_distance['C'] | asteroid_distance['VC']) & theta_delta['PL'], ship_thrust['PS']),
            ctrl.Rule((asteroid_distance['F'] | asteroid_distance['VF'] | asteroid_distance['M']) & theta_delta['NL'], ship_thrust['Z']),
            ctrl.Rule((asteroid_distance['F'] | asteroid_distance['VF'] | asteroid_distance['M']) & theta_delta['NS'], ship_thrust['PS']),
            ctrl.Rule((asteroid_distance['F'] | asteroid_distance['VF'] | asteroid_distance['M']) & theta_delta['Z'], ship_thrust['PL']),
            ctrl.Rule((asteroid_distance['F'] | asteroid_distance['VF'] | asteroid_distance['M']) & theta_delta['PS'], ship_thrust['PS']),
            ctrl.Rule((asteroid_distance['F'] | asteroid_distance['VF'] | asteroid_distance['M']) & theta_delta['PL'], ship_thrust['Z']),
            ctrl.Rule(ship_speed['PL'] & asteroid_distance['M'], ship_thrust['NL']),
            ctrl.Rule(ship_speed['PS'] & (asteroid_distance['M'] | asteroid_distance['C']), ship_thrust['PL']),
            ctrl.Rule(ship_speed['Z'] & (asteroid_distance['C'] | asteroid_distance['VC']), ship_thrust['NS']),
            ctrl.Rule(ship_speed['NL'] & (asteroid_distance['C'] | asteroid_distance['M']), ship_thrust['NL']),
            ctrl.Rule(ship_speed['NS'] & (asteroid_distance['F'] | asteroid_distance['VF']), ship_thrust['PS']),
            ctrl.Rule(ship_speed['PL'] & asteroid_distance['F'], ship_thrust['PL']),
            ctrl.Rule(ship_speed['Z'] & asteroid_distance['M'], ship_thrust['Z']),
            ctrl.Rule(ship_speed['NS'] & asteroid_distance['C'], ship_thrust['NS']),
            ctrl.Rule(ship_speed['NL'] & asteroid_distance['F'], ship_thrust['NL']),
            ctrl.Rule(ship_speed['PS'] & asteroid_distance['C'], ship_thrust['NL']),
            ctrl.Rule(ship_speed['PL'] & asteroid_distance['M'], ship_thrust['PL']),
            ctrl.Rule(ship_speed['PS'] & asteroid_distance['F'], ship_thrust['PS']),
            ctrl.Rule(ship_speed['Z'] & asteroid_distance['C'], ship_thrust['NL']),
        ])

        mine_cond2 = ctrl.Rule(mine_approach['F'] | ship_speed['CM'] | ship_speed['SM'], mine_fire['N'])
        mine_cond1 = ctrl.Rule((mine_approach['L'] | mine_approach['S']) & ship_speed['EM'], mine_fire['Y'])
        #mine_cond2 = ctrl.Rule(mine_thrust['EF'] & mine_approach['F'] & ship_speed['SM'], mine_fire['Y'])

        self.mine_control = ctrl.ControlSystem()
        self.mine_control.addrule(mine_cond1)
        self.mine_control.addrule(mine_cond2)

        # Declare the fuzzy controller, add the rules
        # This is an instance variable, and thus available for other methods in the same object. See notes.
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)
        
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?

        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity.
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.
        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0] # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None
        next_imminent_hit = None
        hits_from_front = 0
        hits_from_back = 0
        curr_dist_back_sum = 0
        curr_dist_front_sum = 0
        front = None
        total_imminent_hits = 0
        total_asteroids = 0

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            a_ship_x = ship_pos_x - a["position"][0]
            a_ship_y = ship_pos_y - a["position"][1]
            a_direction = math.atan2(a["velocity"][1], a["velocity"][0]) # Velocity is a 2-element array [vx,vy].
            a_vel = math.sqrt(a["velocity"][0]**2 + a["velocity"][1]**2)
            a_ship_theta = math.atan2(a_ship_y,a_ship_x)

            #declare a theta...when ship asteroid direction and asteroid direction is within this hit_theta, it means the asteroid will hit
            try:
                hit_theta = abs(math.asin((ship_state["radius"] + a["radius"])/curr_dist))
            except:
                hit_theta = 0
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
            

            if curr_dist <= 150:
                if total_asteroids < 10:
                    total_asteroids += 1
                if ((-math.pi/2 <= a_ship_theta - ship_state['heading'] <= math.pi/2) or (-2*math.pi <= a_ship_theta - ship_state['heading'] <= -3*math.pi/2) or (3*math.pi/2 <= a_ship_theta - ship_state['heading'] <= 2*math.pi)):
                    hits_from_back += 1
                    curr_dist_back_sum += curr_dist
                    front = False
                elif ((-3*math.pi/2 < a_ship_theta - ship_state['heading'] < -math.pi/2) or (math.pi/2 < a_ship_theta - ship_state['heading'] < 3*math.pi/2)):
                    hits_from_front += 1 
                    curr_dist_front_sum += curr_dist
                    front = True

            if (((-hit_theta < (a_direction - a_ship_theta)) and ((a_direction - a_ship_theta) < hit_theta)) or
            ((-hit_theta < (a_ship_theta - a_direction)) and ((a_ship_theta - a_direction) < hit_theta))):
                time = 0
                if (a_ship_theta - a_direction > 0):
                    time = (curr_dist * abs(math.cos(a_ship_theta - a_direction)))/a_vel
                else:
                    time = (curr_dist * abs(math.cos(a_direction - a_ship_theta)))/a_vel
                
                total_imminent_hits += 1
                if next_imminent_hit is None :
                    # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                    next_imminent_hit = dict(aster = a, time = time, front = front)
                else:
                    # closest_asteroid exists, and is thus initialized.
                    if next_imminent_hit["time"] > time:
                        # New minimum found
                        next_imminent_hit["aster"] = a
                        next_imminent_hit["time"] = time
                        next_imminent_hit["front"] = front
        # closest_asteroid is now the nearest asteroid object.
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        # and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!


        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/10) + closest_asteroid["aster"]["radius"] * math.cos(asteroid_direction)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/10) + closest_asteroid["aster"]["radius"] * math.sin(asteroid_direction)

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
            
        # And return your four outputs to the game simulation. Controller algorithm complete.
        thust_sim = ctrl.ControlSystemSimulation(self.thrusting_control, flush_after_run=1)
        thust_sim.input["asteroid_distance"] = closest_asteroid["dist"]
        thust_sim.input['ship_speed'] = ship_state['speed']
        thust_sim.input['theta_delta'] = shooting_theta      
        thust_sim.compute()
        thrust = thust_sim.output['ship_thrust']
        
        mining = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        #mining.input['mine_thrust'] = thrust
        speed = abs(ship_state['speed'])
        mining.input['ship_speed'] = speed
        mining.input['mine_approach'] = total_asteroids
        
        mining.compute()
        if mining.output['mine_fire'] > 0:
            drop_mine = True
        else:
            drop_mine = False
        
        self.eval_frames +=1

        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "WhiteDove Controller"
    
'''
This portion of the code is for implementing the genetic algorithm for the WhiteDoveController. This was 
done in Google Colab and the chromosome was determined to be [7, 100, 153, 0, 162, 181] which is the chromosome
used in the WhiteDoveController.

import numpy as np
import EasyGA
import random

def fitness_func(chromosome):
    print(chromosome)
    my_test_scenario = Scenario(name='Test Scenario',
                              num_asteroids=20,
                              ship_states=[
                                  {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                  # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                              ],
                              map_size=(1000, 800),
                              time_limit=60,
                              ammo_limit_multiplier=0,
                              stop_if_no_ammo=False)

    # Define Game Settings
    game_settings = {'perf_tracker': True,
                    'graphics_type': GraphicsType.Tkinter,
                    'realtime_multiplier': 1,
                    'graphics_obj': None,
                    'frequency': 30}

    game = TrainerEnvironment(settings=game_settings)  # no-graphics simulation
    # Evaluate the game
    pre = time.perf_counter()

    # runs the simulation with the gene input
    #[gene.value for gene in chromosome]
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[WhiteDoveController(chromosome)])
    print(f"asteroids hit: {score.teams[0].asteroids_hit}")
    return score.teams[0].asteroids_hit


def create_chromosome_input():
    thrust_max_point = 250
    ship_speed_max_point = 200
    distance_max_threshold = 800
    distance_large_threshold = 400
    distance_small_threshold = 100

    ship_speed_low = np.random.randint(0, (ship_speed_max_point - 10) / 4)
    ship_speed_mid = np.random.randint(ship_speed_low + 1, (ship_speed_max_point) * (0.75))
    ship_speed_high = np.random.randint(ship_speed_mid + 1, ship_speed_max_point)

    thrust_low = np.random.randint(0, (thrust_max_point - 10) / 4)
    thrust_mid = np.random.randint(thrust_low + 1, (thrust_max_point) * (0.75))
    thrust_high = np.random.randint(thrust_mid + 1, thrust_max_point)

    # maintains order of distance threshold
    # asteroid_distance_mid = np.random.randint(distance_small_threshold + 50, (distance_max_threshold - 100) // 2)
    # asteroid_distance_large = np.random.randint(asteroid_distance_mid + 50, distance_max_threshold - 50)

    chromosome = [ship_speed_low, ship_speed_mid, ship_speed_high, thrust_low, thrust_mid, thrust_high]

    return chromosome



# Define the genetic algorithm parameters
def create_best_chromosome():
    ga = EasyGA.GA()
    ga.gene_impl = create_chromosome_input
    ga.chromosome_length = 1
    ga.population_size = 5
    ga.generation_goal = 10
    ga.target_fitness_type = 'max'
    ga.fitness_function_impl = lambda chromosome: fitness_func(chromosome)
    ga.evolve()
    ga.print_best_chromosome()
    return ga.population[0]
    
import os
if os.path.exists("database.db"):
  os.remove("database.db")
create_best_chromosome()
'''