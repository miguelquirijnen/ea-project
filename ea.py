import r0710304
if __name__ == "__main__":
    a = r0710304.r0710304()
    a.optimize("./tourFiles/tour250.csv")

"""
Values to beat:

tour29: simple greedy heuristic 30350                       best = 27154.48839924464
tour100: simple greedy heuristic 272865                     best = 221568.56127564854
tour250: simple greedy heuristic 49889, shorter than 40k?   best = 40681.861702524344
tour500: simple greedy heuristic 122355                     best = 101765.50862028853
tour750: simple greedy heuristic 119156, shorter than 90k?  best = 97470.5441853045
tour1000: simple greedy heuristic 226541                    best = 184721.5718169272
"""
