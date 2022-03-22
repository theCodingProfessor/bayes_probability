# This file only contains variables, no computing
# the naming convention is as follows:
# r = rise
# d = drop
# L = Low volume traded (less than 15 million shares)
# M = Moderate volume traded (15 - 25 million shares)
# H = Heavy volume traded (35 - 45 million shares)
# X = Extrodinary volument traded (+45 million shares)

# variable name patterns are as follows
# (volume,3-day-prior,2-day-prior,yesterday,today,predicted_label)
# so: L_r_r_r_r_r = means
    # Low volume (L)
    # three-days ago = rise
    # two-days ago = rise
    # yesterday = rise
    # today  = rise
    # tomorrow predicted = rise
    # = (percent indicates confidence in prediction)
    # So this would read the stock has risen for four days
    #  in a row and there is a 10-percent chance that it will
    #  rise again tomorrow.

Some notes on values:
# Values of .1 or ten-percent, 10%
# The minimum prediction values is .1 (10-percent) This always indicates
#  that the occurrence has never been seen in the test data.

# Values of .2, or twenty-percent, 20%
# This indicates that although the occurrence has been seen once in the test data
#  there is not any real confidence that this indicates a trend

# Values of .3, or thirty-percent, 30% (or greater)
# This indicates that the occurrence has been identifid multiple times in the
#  test data, and that while the predictive factor is low for this, it is
#  most often accompanied by a high-value (negated condition). For example
#  where .3 for rise is predicted, a .7 for drop will also occur.
#  This value (.3 or 30% can also occur for values which appear multiple times
#  (more than two times) but there is not enough data to draw a strong
#  conclusion that there is a trend.

# Values higher than .3 (thirty percent are calculated as a percent of their
# total presence in the data set: sum= x+y/2

# The maximum value any occurence is given is .9, or ninty-percent, 90%

#Four rise in a row: (7:7) even odds for next day
L_r_r_r_r_r = .1
L_r_r_r_r_d = .3
M_r_r_r_r_r = .5
M_r_r_r_r_d = .5
H_r_r_r_r_r = .3
H_r_r_r_r_d = .1
X_r_r_r_r_r = .2
X_r_r_r_r_d = .1

#Four drop in a row: (7:2) to rise on next day
L_d_d_d_d_r = .7
L_d_d_d_d_d = .2
M_d_d_d_d_r = .7
M_d_d_d_d_d = .2
H_d_d_d_d_r = .1
H_d_d_d_d_d = .1
X_d_d_d_d_r = .2
X_d_d_d_d_d = .1

#Three rise in a row: (7:13) to rise on next day
L_d_r_r_r_r = .3
L_d_r_r_r_d = .1
M_d_r_r_r_r = .3
M_d_r_r_r_d = .7
H_d_r_r_r_r = .3
H_d_r_r_r_d = .6
X_d_r_r_r_r = .1
X_d_r_r_r_d = .1

#Three drop in a row: (2:6) to rise on next day
L_r_d_d_d_r = .1
L_r_d_d_d_d = .2
M_r_d_d_d_r = .2
M_r_d_d_d_d = .7
H_r_d_d_d_r = .1
H_r_d_d_d_d = .2
X_r_d_d_d_r = .2
X_r_d_d_d_d = .2

#Two rise in a row (21:13)
#  sub rd(rr) (11:10)
L_r_d_r_r_r = .1
L_r_d_r_r_d = .1
M_r_d_r_r_r = .6
M_r_d_r_r_d = .5
H_r_d_r_r_r = .3
H_r_d_r_r_d = .3
X_r_d_r_r_r = .2
X_r_d_r_r_d = .2
#  sub dd(rr) (10:3)
L_d_d_r_r_r = .9
L_d_d_r_r_d = .1
M_d_d_r_r_r = .6
M_d_d_r_r_d = .4
H_d_d_r_r_r = .3
H_d_d_r_r_d = .1
X_d_d_r_r_r = .1
X_d_d_r_r_d = .1

#Two drop in a row (18:8)
#  sub rr(dd) (10:3)
L_r_r_d_d_r = .3
L_r_r_d_d_d = .1
M_r_r_d_d_r = .8
M_r_r_d_d_d = .2
H_r_r_d_d_r = .1
H_r_r_d_d_d = .2
X_r_r_d_d_r = .2
X_r_r_d_d_d = .1
X_r_r_r_d_r = .1
X_r_r_r_d_d = .1
#  sub dr(dd) (9:5)
L_d_r_d_d_r = .1
L_d_r_d_d_d = .1
M_d_r_d_d_r = .6
M_d_r_d_d_d = .4
H_d_r_d_d_r = .1
H_d_r_d_d_d = .2
X_d_r_d_d_r = .3
X_d_r_d_d_d = .2
X_d_r_r_d_r = .2
X_d_r_r_d_d = .1

#Alternate rise_drop_rise (21:20)
#  sub r(rdr) (14:7)
L_r_r_d_r_r = .8
L_r_r_d_r_d = .2
M_r_r_d_r_r = .5
M_r_r_d_r_d = .5
H_r_r_d_r_r = .8
H_r_r_d_r_d = .2
X_r_r_d_r_r = .2
X_r_r_d_r_d = .1
#  sub d(rdr) (7:13)
L_d_r_d_r_r = .1
L_d_r_d_r_d = .3
M_d_r_d_r_r = .4
M_d_r_d_r_d = .6
H_d_r_d_r_r = .1
H_d_r_d_r_d = .2
X_d_r_d_r_r = .1
X_d_r_d_r_d = .2

#Alternate rise_rise_drop (21:13)
#  sub r(rrd) (11:9)
L_r_r_r_d_r = .2
L_r_r_r_d_d = .1
M_r_r_r_d_r = .5
M_r_r_r_d_d = .5
H_r_r_r_d_r = .3
H_r_r_r_d_d = .2
X_r_r_r_d_r = .1
X_r_r_r_d_d = .1
#  sub d(rrd) (10:4)
L_d_r_r_d_r = .2
L_d_r_r_d_d = .1
M_d_r_r_d_r = .6
M_d_r_r_d_d = .4
H_d_r_r_d_r = .7
H_d_r_r_d_d = .2
X_d_r_r_d_r = .2
X_d_r_r_d_d = .1

#Alternate drop_rise_drop (20:13)
#  sub r(drd) (13:7)
L_r_d_r_d_r = .5
L_r_d_r_d_d = .5
M_r_d_r_d_r = .7
M_r_d_r_d_d = .3
H_r_d_r_d_r = .1
H_r_d_r_d_d = .1
X_r_d_r_d_r = .2
X_r_d_r_d_d = .1
#  sub d(drd) (7:6)
L_d_d_r_d_r = .1
L_d_d_r_d_d = .1
M_d_d_r_d_r = .6
M_d_d_r_d_d = .4
H_d_d_r_d_r = .5
H_d_d_r_d_d = .5
X_d_d_r_d_r = .1
X_d_d_r_d_d = .1

#Alternate drop_drop_rise (15:13)
#  sub r(ddr) (10:10)
L_r_d_d_r_r = .5
L_r_d_d_r_d = .5
M_r_d_d_r_r = .5
M_r_d_d_r_d = .5
H_r_d_d_r_r = .5
H_r_d_d_r_d = .5
X_r_d_d_r_r = .2
X_r_d_d_r_d = .2
#  sub d(ddr) (5:3)
L_d_d_d_r_r = .3
L_d_d_d_r_d = .1
M_d_d_d_r_r = .2
M_d_d_d_r_d = .7
H_d_d_d_r_r = .2
H_d_d_d_r_d = .1
X_d_d_d_r_r = .2
X_d_d_d_r_d = .1
