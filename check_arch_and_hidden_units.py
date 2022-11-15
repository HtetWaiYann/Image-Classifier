def check_arch_and_hidden_units(arch, hidden_units):
    valid = True
    if arch == 'alexnet':
        if hidden_units >= 9216:
            print('Hidden units for Alexnet Architecture cannot be greater than 9216')
            valid = False
        elif hidden_units <= 102:
            print('Hidden units for Alexnet Architecture cannot be less than 102')
            valid = False
        
    elif arch == 'densenet':
        if hidden_units >= 1024:
            print('Hidden units for Alexnet Architecture cannot be greater than 1024')
            valid = False
        elif hidden_units <= 102:
            print('Hidden units for Alexnet Architecture cannot be less than 102')
            valid = False
    else:
        print('The',arch,'architecture is not supported.')
        valid = False
    return valid