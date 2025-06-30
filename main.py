import sys
from examples.multi_class import multi_class_example
from examples.binary_class import binary_class_example
from examples.regression import regression_example


def main(model_example: str):
    if model_example == 'binary_class':
        return binary_class_example()
    elif model_example == 'multi_class':
        return multi_class_example()
    elif model_example == 'regression':
        return regression_example()
    else:
        raise ValueError('model_example must be binary_class, multi_class or regression')

if __name__ == '__main__':
    selected_model_example = sys.argv[1]
    main(model_example=selected_model_example)
