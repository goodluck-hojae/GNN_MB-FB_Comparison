import re
import ast


def parse_log(log_text, sampler):
    tuple_pattern = r"\('([^']+)'\s*,\s*'([^']+)'.*?\)"

    tuple_match = re.search(tuple_pattern, log_text)
    
    if tuple_match:
        tuple_start, tuple_end = tuple_match.span()
        tuple_line = log_text[tuple_start:tuple_end]
        try:
            tuple_data = ast.literal_eval(tuple_line)
        except Exception as e:
            print(f"Error converting tuple: {e}")
            return None
        dataset = tuple_data[0]  # First element
        model = tuple_data[1]  # Second element

        epoch_time = tuple_data[-7]
        converge_epoch = tuple_data[-4]

        accuracy_pattern = r'Test Accuracy ([\d.]+)'
        accuracies = re.findall(accuracy_pattern, log_text)
        times = [epoch_time] * len(accuracies)

        accuracies = [float(a) for a in accuracies]

        return dataset, model, times[:converge_epoch], accuracies[:converge_epoch], sampler
    else:
        return None