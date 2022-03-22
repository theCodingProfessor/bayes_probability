# Clinton Garwood
# Code Discussion on Probability
# CSCE 364 Spring 2022
# Independent, Dependent and Bayes

import random

# # Probability of Event 1 (independent)
# CF1_Head = .5
# CF1_Tail = .5
# CF2_Head = .5
# CF2_Tail = .5
# one_coin_odds = [CF1_Head,CF1_Tail]
# two_coin_odds=[CF1_Head,CF1_Tail,CF2_Head,CF2_Tail]
#
# CF1 = ['heads', 'tails']
# def single_coin_toss(single_coin):
#     print("\n\tThrow the Coin with Independent Odds:")
#     print("\tA single toss of the coin has 50/50 odds, this time it is", single_coin)
#     return
# single_coin_toss(CF1[random.randint(0, 1)])
#
# def get_single_toss(CF1):
#     return CF1[random.randint(0, 1)]
#
# def compute_one_coin_odds(one_coin_odds):
#     print("\n\tOne Throw of the Coin (Compute Independent Odds:")
#     print("\tThe odds for Heads are P(CF1=Heads) = ", one_coin_odds[0])
#     print("\tThe odds for Tails are P(CF1=Tails) = ", one_coin_odds[1])
#     return
# compute_one_coin_odds(one_coin_odds)
#
# def compute_two_toss_odds(tco):
#     #two_coin_odds = [CF1_Head, CF1_Tail, CF2_Head, CF2_Tail]
#     print("\n\tTwo Throws of the Coin (Compute Independent Odds:")
#     print("\tThe odds for Heads-Heads: P(CF1_h, CF2_h) = P(FC1_h)P(FC2_h) =", tco[0] *tco[2])
#     print("\tThe odds for Heads-Tails: P(CF1_h, CF2_t) = P(FC1_h)P(FC2_t) =", tco[0] *tco[3])
#     print("\tThe odds for Tails-Heads: P(CF1_t, CF2_h) = P(FC1_t)P(FC2_h) =", tco[1] *tco[2])
#     print("\tThe odds for Tails-Tails: P(CF1_t, CF2_t) = P(FC1_t)P(FC2_t) =", tco[1] *tco[3])
#     return
# compute_two_toss_odds(two_coin_odds)
#
# def make_two_throws():
#     # t_1 = get_single_toss(CF1)
#     # t_2 =get_single_toss(CF1)
#     print("\n\tRunning five double-throws")
#     for x in range(5):
#         t_3 = get_single_toss(CF1)
#         t_4 = get_single_toss(CF1)
#         print("\tTwo-throw result {} was {}, {}".format(x,t_3,t_4))
# make_two_throws()
#
# prob_a = [.5,'Heads']
# prob_b = [1,'Heads']
#
# def bayes_formula_indi(prob_a, prob_b):
#     # P(CF2 | CF1) = P(CF2, CF1)/P(CF1)
#     prob_ab = ( (prob_a[0] * prob_b[0])/prob_b[0] )
#     print("\n\tUsing Bayes Formula  P(E2 | EV1) = P(EV2, EV1)/P(EV1)")
#     print("\tThe Probability for Coin 2 to be {} given Coin 1 is {}, is {}".format(prob_b[1],prob_a[1],prob_ab))
#     return prob_ab
# bayes_formula_indi(prob_a,prob_b)
#
# prob_a = [.5,'Heads']
# prob_b = [1,'Heads']
# def bayes_formula(prob_a, prob_b):
#     # P(CF2 | CF1) = P(CF2, CF1)/P(CF1)
#     prob_ab = ( (prob_a[0] * prob_b[0])/prob_b[0] )
#     print("\n\tUsing Bayes Formula  P(E2 | EV1) = P(EV2, EV1)/P(EV1)")
#     print("\tThe Probability for Coin 2 to be {} given Coin 1 is {}, is {}".format(prob_b[1],prob_a[1],prob_ab))
#     return prob_ab
# bayes_formula(prob_a,prob_b)
#
# def return_from_bayes_formula(prob_a, prob_b):
#     prob_ab = ( (prob_a * prob_b)/prob_b )
#     return prob_ab
#
# two_throw_outcomes = ['Heads-Heads', 'Heads-Tails', 'Tails-Heads','Tails-Tails']
# single_odds = [['CF1_Head',.5],['CF1_Tail',.5],['CF2_Head',.5],['CF2_Tail',.5]]
# first_coin_results = ['head','tail']
# def two_throw_dep_odds():
#     first_toss = get_single_toss(CF1)
#     if first_toss == 'tail':
#         print("\n\tThe coin toss landed as:", first_toss, "therefore:")
#         print("\tThe odds of", two_throw_outcomes[2], "are", return_from_bayes_formula(single_odds[2][1],1))
#         print("\tThe odds of", two_throw_outcomes[3], "are", return_from_bayes_formula(single_odds[3][1],1))
#         print("\tNeither {} or {} are possible outcomes".format(two_throw_outcomes[0],two_throw_outcomes[1]))
#     else:
#         print("\n\tThe coin toss landed as:", first_toss, "therefore:")
#         print("\tThe odds of", two_throw_outcomes[0], "are", return_from_bayes_formula(single_odds[0][1],1))
#         print("\tThe odds of", two_throw_outcomes[1], "are", return_from_bayes_formula(single_odds[1][1],1))
#         print("\tNeither {} or {} are possible outcomes".format(two_throw_outcomes[2],two_throw_outcomes[3]))
#     return
# two_throw_dep_odds()
#
# stop_light_colors = ['red','green']
# SL1_Green = .5
# SL1_Red = .5
# SL2_Green = .5
# SL2_Red = .5
# one_light_outcomes = [SL1_Green,SL1_Red]
# two_light_outcomes=[SL1_Green,SL1_Red,SL2_Green,SL2_Red]
# one_green_probabilities = [.5,.5,.5]
# one_green_outcomes=["Red-Green","Green-Red","Green-Green"]


def bayes_proper(L1,L2,outcomes):
    prob_ab = ( ((L1 * L2))/(outcomes) /L2 )
    return prob_ab
print(bayes_proper(1,.5,3))

# def bayes_formula_broken(prob_a, prob_b):
#     # P(CF2 | CF1) = P(CF2, CF1)/P(CF1)
#     prob_ab = ( (prob_a * prob_b)/prob_b )
#     return prob_ab
#
# print("\n\tThe probability reported by Bayes for Green-Green, given SL1_Green is:", bayes_formula_broken(SL2_Green,1))
# print("\tBut it should be .333 or one-third.")
# print("\tTo correct this, we should correct Bayes Formula to accept a total number")
# print("\tof possible solutions, and calculate P(A|B) / number_of_possibilities")
#
# def bayes_formula_fixed(prob_a, prob_b,prob_list):
#     # P(CF2 | CF1) = P(CF2, CF1)/P(CF1)
#     total_prob = sum(prob_list)
#     prob_ab = ( (prob_a * prob_b)/prob_b )/total_prob
#     return prob_ab
# print("\n\tRunning the updated Probability for Bayes for Green-Green, given SL1_Green is: {:.3f}".format(bayes_formula_fixed(SL2_Green,1,one_green_probabilities)))
#
# print("\n\tAlgebraic extensions of Bayes Formula:")
# print("\tP(E,F) = P(E)*P(F)")
# print("\tP(E|F) = P(E,F) /P(F)")
# print("\t		== P(E)*P(F)/P(F)")
# print("\tP(E,F) = P(E|F)*P(F)")
# print("\t		== (P(E,F)/P(F))*P(F)")
# print("\t		== (P(E)*P(F) ) / P(F) ) * P(F)")
#
# print("\n\tWhere E=.5 and F=1")
# print("\tP(E,F) = P(E)*P(F) = .5 * 1 == ", .5 * 1)
# print("\tP(E|F) = P(E,F) /P(F)")
# print("\t		== P(E)*P(F)/P(F) = (.5 * 1)/1 == ", (.5 * 1)/1)
# print("\tP(E,F) = P(E|F)*P(F) = (.5 * 1)*1 == ", (.5 * 1)*1)
# print("\t		== (P(E,F)/P(F))*P(F)")
# print("\t		=== (P(E)*P(F) ) / P(F) ) * P(F) = (.5 * 1)/1 ==", (.5 * 1)/1)
#
