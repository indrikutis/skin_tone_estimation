import os
import re
import pandas as pd


def extract_metrics_from_file(file_path):
    """Extracts the relevant metrics from a txt file."""
    with open(file_path, "r") as file:
        content = file.read()

    # Regular expressions to extract the values
    train_loss = re.search(r"Train loss:\s*(\d+\.\d+)", content)
    train_accuracy = re.search(r"Train accuracy:\s*(\d+\.\d+)", content)
    test_loss = re.search(r"Test loss:\s*(\d+\.\d+)", content)
    test_accuracy = re.search(r"Test accuracy:\s*(\d+\.\d+)", content)

    # If the value is found, extract it, otherwise set it as None
    train_loss = float(train_loss.group(1)) if train_loss else None
    train_accuracy = float(train_accuracy.group(1)) if train_accuracy else None
    test_loss = float(test_loss.group(1)) if test_loss else None
    test_accuracy = float(test_accuracy.group(1)) if test_accuracy else None

    return train_loss, train_accuracy, test_loss, test_accuracy


def extract_metrics_from_directory(directory_path):
    """Extracts metrics from all txt files in a directory."""
    metrics = {}

    # Traverse the directory and get all txt files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                train_loss, train_accuracy, test_loss, test_accuracy = (
                    extract_metrics_from_file(file_path)
                )

                # Use the filename as the key for each metric
                metrics[file] = {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }

    return metrics


def compare_metrics_across_folders(directories):
    """Compares the metrics across multiple directories and returns a structured table."""
    all_metrics = {}

    for directory in directories:
        metrics = extract_metrics_from_directory(directory)
        all_metrics[directory] = metrics

    # Create a list of rows, where each row is a dictionary of values for a specific file
    rows = []

    for directory, metrics in all_metrics.items():
        for file, values in metrics.items():
            folder_name = directory.split("/")[-1]

            row = {
                "file_name": file,
                "train_loss": values["train_loss"],
                "train_accuracy": values["train_accuracy"],
                "test_loss": values["test_loss"],
                "test_accuracy": values["test_accuracy"],
                "directory": directory,
            }
            rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.pivot_table(
        index="file_name",
        columns="directory",
        values=["train_loss", "train_accuracy", "test_loss", "test_accuracy"],
        aggfunc="first",
    )

    comparison_df.columns = [
        os.path.basename(col[1]) + f"_{col[0]}" for col in comparison_df.columns
    ]

    return comparison_df


def extract_model_summaries(directories):

    model_data = {}

    # Iterate over the directories provided
    for directory in directories:
        if os.path.isdir(directory):
            model_data[directory] = {}
            for root, dirs, files in os.walk(directory):
                for file in files:
                    index = 1
                    file_name = os.path.basename(file)

                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            content = f.read()

                            pattern = r"Model:\s*(.*?)\s*Train accuracy:\s*([0-9.]+)\s*Test accuracy:\s*([0-9.]+)"
                            matches = re.findall(pattern, content)

                            for model, train_accuracy, test_accuracy in matches:
                                if directory not in model_data:
                                    model_data[directory] = {}

                                if file_name not in model_data[directory]:
                                    model_data[directory][file_name] = {}

                                if model not in model_data[directory][file_name]:
                                    model_data[directory][file_name][model] = {
                                        "train": [],
                                        "test": [],
                                    }
                                else:
                                    model = model + str(index)
                                    model_data[directory][file_name][model] = {
                                        "train": [],
                                        "test": [],
                                    }
                                    index = index + 1

                                # Append the accuracies
                                model_data[directory][file_name][model]["train"].append(
                                    float(train_accuracy)
                                )
                                model_data[directory][file_name][model]["test"].append(
                                    float(test_accuracy)
                                )

    directory_tables = {}

    for directory, file_data in model_data.items():
        combined_data = {}
        file_names = list(file_data.keys())

        for file_name, models in file_data.items():
            for model, accuracies in models.items():
                train_col = f"{model}_train"
                test_col = f"{model}_test"

                # Initialize columns for train/test if not already present
                if train_col not in combined_data:
                    combined_data[train_col] = [None] * len(file_names)
                if test_col not in combined_data:
                    combined_data[test_col] = [None] * len(file_names)

                # Find the index of the file and append the accuracy values
                file_index = file_names.index(file_name)
                combined_data[train_col][file_index] = accuracies["train"][
                    0
                ] 
                combined_data[test_col][file_index] = accuracies["test"][
                    0
                ]  

        combined_data["file_name"] = file_names
        directory_df = pd.DataFrame(combined_data)
        directory_tables[directory] = directory_df

        all_dataframes = []

        # Iterate through directory_tables and process each DataFrame
        for directory, directory_df in directory_tables.items():

            directory_df["directory"] = directory
            all_dataframes.append(directory_df)

        # Concatenate all DataFrames into one
        comparison_df = pd.concat(all_dataframes, ignore_index=True)

    return comparison_df
