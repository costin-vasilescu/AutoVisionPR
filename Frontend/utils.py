import numpy as np

with open('./counties.txt', 'r') as f:
    counties = f.read().replace(' ', '').split(',')
counties.remove('B')


def count_digits(string):
    return sum(c.isdigit() for c in string)


def count_letters(string):
    return sum(c.isupper() for c in string)


def match_plate_text(string):
    score = 0

    if string[0] == 'B' and not string[1].isupper():
        county_score = 1

        # Provisional Plate
        provisional_score = count_digits(string[1:6]) / 5

        # Normal plate
        part1_score1 = count_digits(string[1:4]) / 3
        part2_score1 = count_letters(string[4:7]) / 3
        extra_score1 = len(string[7:])

        part1_score2 = count_digits(string[1:3]) / 2
        part2_score2 = count_letters(string[3:6]) / 3
        extra_score2 = len(string[6:])

        score = np.average([county_score,
                            np.max([provisional_score, (part1_score1 + part2_score1) / 2 - 0.1 * extra_score1,
                                    (part1_score2 + part2_score2) / 2 - 0.1 * extra_score2])])
    elif string[:2].isupper():
        county_score = 1 if string[:2] in counties else 0.5

        # Provisional Plate
        provisional_score = count_digits(string[2:7]) / 5

        # Normal plate
        part1_score = count_digits(string[2:4]) / 2
        part2_score = count_letters(string[4:7]) / 3
        extra_score = len(string[7:])

        score = np.average(
            [county_score, np.max([provisional_score, (part1_score + part2_score) / 2 - 0.1 * extra_score])])

    return score
