# Demo on Bayes Formula
# calculations and plotting of ranges

import math
import matplotlib.pyplot as plt
import numpy as np

# Generic Bayes Formula
def compute_bayes(p_1, p_2_on_1, p_2):
    probability = (p_1 * p_2_on_1)/p_2
    return probability

def smoke_on_fire(p_f, p_s_f, p_s):
    p_f_s = (p_f * p_s_f)/p_s
    return p_f_s

fire = .01
s_f = .90
smoke = .2
PFS = smoke_on_fire(fire, s_f,smoke)
#print("The probability of fire when smoke is nearby is {:.2f}".format(PFS))


def compute_fire_ranges(fires, s_f, smoke):
    chance_of_fire = fires[1]
    smoke_and_fire = list()
    for each, value in enumerate(smoke):
        F_S = compute_bayes(chance_of_fire, s_f, smoke[each])
        smoke_and_fire.append(F_S)
    xpoints = smoke_and_fire
    ypoints = fires
    plt.plot(xpoints, ypoints)
    plt.xlabel("Chance of Fire (percent)")
    plt.ylabel("Chance of Smoke")
    plt.show()
    return

fires = [.01,.05,.1,.15,.2,.25,.30,.35,.4]
s_f = .90
smoke = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
#compute_fire_ranges(fires, s_f, smoke)


# What happens when variables change?
rain_array = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
cloudy_array = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
cloud_on_rain = [.1,.2,.3,.4,.5,.6,.7,.8,.9]

# What is the chance that rain will occur
# P(Rain|Cloud) =  P(Rain) P(Cloud|Rain)/P(Cloud)
# P(Rain|Cloud) =  0.1 x 0.5/0.4   = .125
C_R = compute_bayes(rain_array[0],cloud_on_rain[4],cloudy_array[3])
print("/nP(Rain) P(Cloud|Rain)/P(Cloud) as 0.1 x 0.5/0.4 = {:.2f}".format(C_R))

# Graph the probability of rain given the full range of cloudiness (cloud_on_rain)

def compute_cloud_range(rain_array, cloud_on_rain, cloudy_array):
    p_r = rain_array[0]
    p_c_r = cloud_on_rain[4]
    cloud_range = list()
    for each, value in enumerate(cloudy_array):
        R_C = compute_bayes(p_r, p_c_r, cloudy_array[each])
        cloud_range.append(R_C)
    xpoints = rain_array
    ypoints = cloud_range
    plt.plot(xpoints, ypoints)
    plt.xlabel("Chance of Clouds")
    plt.ylabel("Chance of Rain (percent)")
    plt.show()

#compute_cloud_range(rain_array, cloud_on_rain, cloudy_array)

def compute_rain_range(rain_array, cloud_on_rain, cloudy_array):
    p_c_r = cloud_on_rain[4]
    p_c = cloudy_array[3]
    rain_range = list()
    for each, value in enumerate(rain_array):
        R_C = compute_bayes(rain_array[each], p_c_r, p_c)
        rain_range.append(R_C)
    xpoints = rain_array
    ypoints = rain_range
    plt.plot(xpoints, ypoints)
    plt.xlabel("Rain Range")
    plt.ylabel("Chance of Rain (percent)")
    plt.show()

# compute_rain_range(rain_array, cloud_on_rain, cloudy_array)


def compute_rain_ranges(rain_array, cloud_on_rain, cloudy_array):
    p_c_r = cloud_on_rain[4]
    p_c_3 = cloudy_array[3]
    rain_range = list()
    ypoints = [9,8,7,6,5,4,3,2,1]
    plt.plot([compute_bayes(rain_array[x], p_c_r, cloudy_array[3]) for x,y in enumerate(rain_array)],rain_array,'-',label='30% cloudy')
    plt.plot([compute_bayes(rain_array[x], p_c_r, cloudy_array[4]) for x,y in enumerate(rain_array)],rain_array,'--',label='40% cloudy')
    plt.plot([compute_bayes(rain_array[x], p_c_r, cloudy_array[5]) for x,y in enumerate(rain_array)],rain_array,':',label='50% cloudy')
    plt.plot([compute_bayes(rain_array[x], p_c_r, cloudy_array[6]) for x,y in enumerate(rain_array)],rain_array,'.',label='60% cloudy')
    plt.legend()
    plt.xlabel("Bayes Probability")
    plt.ylabel("Chance of Rain (percent)")
    plt.show()

compute_rain_ranges(rain_array, cloud_on_rain, cloudy_array)




