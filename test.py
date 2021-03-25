def five(e):
    return 5
def six(e):
    return 6

sample = [six, five, six]
func = [x(5) for x in sample]
print(func)
