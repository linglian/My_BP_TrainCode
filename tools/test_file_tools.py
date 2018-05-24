import file_tools

if __name__ == '__main__':
    import sys
    import getopt
    path = None
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
    for op, value in opts:
        if op == '-f':
            path = value
    
    def display(things):
        print file_tools.getFileName(things)
    if path is not None:
        file_tools.traverse_floder(path, display, '.py')
    else:
        print('Please Enter Path')