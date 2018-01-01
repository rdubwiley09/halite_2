import math

#Add two vectors of the form [magnitude,angle] (in degrees)
def add_vectors(vector1, vector2):
    vector1_radians = (vector1[1]/180)*math.pi
    vector2_radians = (vector2[1]/180)*math.pi
    vector1_x = vector1[0]*math.cos(vector1_radians)
    vector1_y = vector1[0]*math.sin(vector1_radians)
    vector2_x = vector2[0]*math.cos(vector2_radians)
    vector2_y = vector2[0]*math.sin(vector2_radians)
    add_x = vector1_x + vector2_x
    add_y = vector1_y + vector2_y
    magnitude = math.sqrt(add_x*add_x+add_y*add_y)
    angle_radians = math.atan2(add_y,add_x)
    angle_degrees = (angle_radians*180)/math.pi
    return [magnitude,angle_degrees]

def resize_vector(vector, resize_constant):
    output = vector
    output[0] = vector[0]*resize_constant
    return output
