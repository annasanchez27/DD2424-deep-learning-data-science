
def load_data(file):
    """
    Loads the data
    :param file: Input file
    :return: Dictionary
    {'data': vector with the data, 'char': vocabulary,
    'K': length of the vocabulary,
    'char_to_ind': character to index,
    'ind_to_char': index to character}
    """
    with open(file) as f:
        lines = open(file, 'r', encoding='utf8').read()
    char = list(set(lines))
    k = len(char)
    char_to_ind = {c: char.index(c) for c in char}
    ind_to_char = {value: key for (key, value) in char_to_ind.items()}
    return {'data': lines, 'char': char, 'K': k,
            'char_to_ind': char_to_ind,
            'ind_to_char': ind_to_char}


lines = load_data("/Users/annasanchezespunyes/Documents/GitHub/DD2424/lab4/data/goblet_book.txt")
