import csv
import numpy as np


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


"""
    This function takes a set of feature expressions and transforms it
    into a dictionary which holds the feature expression as key and
    the respective numeric representation as value.
"""
def map_expressions_to_numbers(feature_expressions: set) -> dict:
    feature_expressions = list(feature_expressions)
    feature_expressions.sort()
    feature_map = {}
    for i, title in enumerate(feature_expressions):
        feature_map[title] = i
    return feature_map


"""
    This function transforms every datapoint from the CSV file into a
    feature vector and a label and returns a list containing all
    feature vectors and labels.

    The model will be trained with feature vectors that look as 
    follows:

    x1: job title (0 - 130)
    x2: experience level (0 - 3)
    x3: employee type (0 - 3)
    x4: work model (0 - 2)
    x5: employee residence (0 - 86)
    x6: company location (0 - 74)
    x7: company size (0 - 2)

    The label y will be the respective salary in USD.
"""
def transform_for_model(rows: list, feature_expressions: dict) -> [list, list]:
    # Create feature expression maps.
    job_title_expression = map_expressions_to_numbers(feature_expressions['job_title'])
    experience_level_expression = map_expressions_to_numbers(feature_expressions['experience_level'])
    employment_type_expression = map_expressions_to_numbers(feature_expressions['employment_type'])
    work_models_expression = map_expressions_to_numbers(feature_expressions['work_models'])
    employee_residence_expression = map_expressions_to_numbers(feature_expressions['employee_residence'])
    company_location_expression = map_expressions_to_numbers(feature_expressions['company_location'])
    company_size_expression = map_expressions_to_numbers(feature_expressions['company_size'])

    # Transform rows.
    feature_vectors = []
    labels = []
    for row in rows[1:]:
        feature_vector = []
        feature_vector.append(job_title_expression[row[0]])
        feature_vector.append(experience_level_expression[row[1]])
        feature_vector.append(employment_type_expression[row[2]])
        feature_vector.append(work_models_expression[row[3]])
        feature_vector.append(employee_residence_expression[row[5]])
        feature_vector.append(company_location_expression[row[9]])
        feature_vector.append(company_size_expression[row[10]])
        feature_vectors.append(feature_vector)
        labels.append([float(row[8])])
    return [feature_vectors, labels]


"""
    This function performs a Z-score normalization onto all the output
    labels. The implementation has the mean and the standard deviation
    of the set of labels hardcoded. The code for how it was retrieved
    is commented out at the beginning of the function.
"""
def scale_labels(labels: list) -> list:
    """
    mean = 0
    stddev = 0

    for label in labels:
        mean += label[0]
    mean = mean / len(labels)
    
    for label in labels:
        stddev = (label[0] - mean) ** 2
    stddev = stddev / len(labels)
    """

    # Perform Z-Score normalization.
    mean = 145560.55856948023
    stddev = 447196.39576477476
    normalized_labels = []
    for label in labels:
        new_label = (label[0] - mean) / stddev
        normalized_labels.append([new_label])
    return normalized_labels


"""
    This function is used to undo the Z-score normalization of the
    models output.
"""
def undo_label_scaling(label: float) -> float:
    mean = 145560.55856948023
    stddev = 447196.39576477476
    return label * stddev + mean


"""
    This function performs a min-max normalization onto all features of
    the feature vectors. The values for min and max can be retrieved 
    from the tables in the README.md file in this directory.
"""
def scale_features(feature_vectors: list) -> list:
    normalized_feature_vectors = []
    for vector in feature_vectors:
        normalized_vector = []
        normalized_vector.append(vector[0] / 131)
        normalized_vector.append(vector[1] / 3)
        normalized_vector.append(vector[2] / 3)
        normalized_vector.append(vector[3] / 2)
        normalized_vector.append(vector[4] / 86)
        normalized_vector.append(vector[5] / 74)
        normalized_vector.append(vector[6] / 2)
        normalized_feature_vectors.append(normalized_vector)
    return normalized_feature_vectors


"""
    This function can be called to immediately retrieve feature vectors
    and labels for model training.
"""
def provide_data_for_model() -> [list, list]:
    csv_rows = extract_data()
    feature_expressions = analyze_features(
        rows=csv_rows
    )
    unscaled_vectors = transform_for_model(
        rows=csv_rows, 
        feature_expressions=feature_expressions
    )
    scaled_feature_vectors = scale_features(
        feature_vectors=unscaled_vectors[0]
    )
    scaled_labels = scale_labels(
        labels=unscaled_vectors[1]
    )
    return [scaled_feature_vectors, scaled_labels]


if __name__ == '__main__':
    result = provide_data_for_model()

    for i, vector in enumerate(result[0]):
        print(vector, result[1][i])