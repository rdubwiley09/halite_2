from math import exp

def partial_derivative(point_x,object_x,other_point,other_object,width_weight,slope_constant):
    return slope_constant*2*width_weight*(object_x-point_x)*exp(-1*width_weight*(pow(object_x-point_x,2)+pow(other_object-other_point,2)))

def compute_gradient(point_x,point_y,object_x,object_y,width_weight,slope_constant):
    x_partial = partial_derivative(point_x,object_x,point_y,object_y,width_weight,slope_constant)
    y_partial = partial_derivative(point_y,object_y,point_x,object_y,width_weight,slope_constant)
    return x_partial, y_partial
