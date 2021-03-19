

def is_sequence(object):
    try:
        iter(object)
        return True
    except TypeError as te:
        return False