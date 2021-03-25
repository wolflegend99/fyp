#
# random stuff goes here
#

def five(e):
    return 5 + e
def six(e):
    return 6 + e

sample = [six, five, six]
func = [x(5) for x in sample]
print(func)
