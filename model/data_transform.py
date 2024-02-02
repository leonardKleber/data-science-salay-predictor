import csv


"""
    This function extracts the data from the CSV dataset file and
    converts it into a list, containing all rows as lists of strings.
"""
def extract_data() -> list:
    rows = []
    with open('../dataset.csv', newline='') as file:
        for row in file:
            row_list = row.split(',')
            row_list[len(row_list)-1] = row_list[len(row_list)-1].strip()
            rows.append(row_list)
    return rows


"""
    This function returns a dictionary containing every feature with a
    set of each features possible expressions.

    It takes a list of rows as input as generates by the extract_data()
    function.
"""
def analyze_features(rows: list) -> dict:
    feature_index = {}
    feature_expressions = {}

    for i, element in enumerate(rows[0]):
        feature_index[i] = element
        feature_expressions[element] = set()

    for row in rows[1:]:
        for i, element in enumerate(row):
            feature_expressions[feature_index[i]].add(element)

    return feature_expressions


if __name__ == '__main__':
    rows = extract_data()

    feature_expressions = analyze_features(rows)