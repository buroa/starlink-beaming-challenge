import sys
import ksat

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if (len(arguments) > 0):
        case = arguments[0]
        ksat.eval(case)
    else:
        print('You did something bad.')