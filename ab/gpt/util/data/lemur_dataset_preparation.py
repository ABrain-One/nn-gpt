import json
import os
import ab.nn.api as api


class DatasetPreparation:
    @staticmethod
    def extract_full_code(code_file):
        """
        Extracts the full source code from the specified file.

        :param code_file: Path to the code file.
        :return: Full code as a string.
        """
        try:
            with open(code_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: File {code_file} not found.")
            return "File not found."


    @staticmethod
    def extract_net_class(code_file):
        """
        Extracts only Net class code from the specified file.

        :param code_file: Path to the code file.
        :return: Net class code as a string.
        """
        in_class = False
        class_code = []

        try:
            with open(code_file, 'r') as f:
                for line in f:
                    # Find the beginning of the Net class
                    if line.strip().startswith("class Net("):
                        in_class = True

                    # Save the lines if we are inside the class
                    if in_class:
                        class_code.append(line)

                    # Stop saving when exiting the class (if we encounter another class definition)
                    if in_class and line.strip() and line.strip().startswith("class ") and not line.strip().startswith(
                            "class Net("):
                        break
        except FileNotFoundError:
            print(f"Warning: File {code_file} not found.")
            return "Net class not found."

        # Combine lines into one code block
        return ''.join(class_code) if class_code else "Net class not found."


    @staticmethod
    def save_as_json(data, filename):
        """
        Saves data to a JSON file.

        :param data: Data to save.
        :param filename: File name.
         """
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    @staticmethod
    def prepare_json_dataset_for_llm_format(input_file_path, output_file_path):
        """
        Loads data from the specified JSON file and saves it to another JSON file with "question" and "answer" fields.

        :param input_file_path: Path to the file containing the data to be processed.
        :param output_file_path: Name of the file to save.
        """

        # Read data from file
        print(f"File: {input_file_path}")
        with open(input_file_path, "r") as f:
            data = json.load(f)

        # Form a new list, where each entry has two columns: "question" and "answer"
        output_data = []
        for entry in data:
            if 'accuracy' in entry and 'prm' in entry:
                hyperparameters = entry['prm']
                prm_names = ", ".join(hyperparameters.keys())
                prm_values = ", ".join([f"{key} = {value}" for key, value in hyperparameters.items()])
                model_code_file = os.path.join(
                    "path/to/nn-dataset/ab/nn/nn", f"{entry['metric']}.py"  # ! Insert the correct path !
                )

                # Extract a model code
                model_full_code = DatasetPreparation.extract_full_code(model_code_file)

                instruction_str = (
                    f"Generate only the values (don't provide any explanation) of the hyperparameters ({prm_names})"
                    f"of a given model: {entry['metric']} for the task: {entry['task']} on dataset: {entry['dataset']}, "
                    f"with transformation: {entry['transform_code']}, so that the model achieves "
                    f"accuracy = {entry['accuracy']} with number of training epochs = {entry['epoch']}. "
                    f"Code of that model:\n {model_full_code}"
                )

                output_str = (
                    f"Here are the hyperparameter values for which the model will achieve the specified accuracy: {prm_values}."
                )

                output_data.append({"question": instruction_str, "answer": output_str})

        # Save to the specified filename
        with open(output_file_path, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Data saved successfully to {output_file_path}")


    @staticmethod
    def add_nn_code_field_to_json(input_file_path, output_file_path):
        """
        Adds the 'nn_code' field to a JSON file by extracting the corresponding model code.

        :param input_file_path: Path to the original JSON file.
        :param output_file_path: Path to save the updated JSON file.
        """
        # Load the JSON file
        with open(input_file_path, "r") as f:
            data = json.load(f)

        # Process each entry
        for entry in data:
            model_code_file = os.path.join(
                "path/to/nn-dataset/ab/nn/nn", f"{entry['metric']}.py" # ! Insert the correct LEMUR Dataset path !
            )
            # Extract model code and add it to the JSON entry
            entry["nn_code"] = DatasetPreparation.extract_full_code(model_code_file)

        # Save the updated JSON file
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Updated JSON saved to {output_file_path}")


    @staticmethod
    def merge_json_files(file_1, file_2, output_file):
        """
        Merges data from two JSON files into one.

        :param file_1: Path to the first JSON file.
        :param file_2: Path to the second JSON file.
        :param output_file: Path to the output JSON file.
        """
        # Load data from the first file
        with open(file_1, "r") as f1:
            data_1 = json.load(f1)

        # Load data from the second file
        with open(file_2, "r") as f2:
            data_2 = json.load(f2)

        # Data Merging
        combined_data = data_2 + data_1

        # Save the merged data to a new file
        with open(output_file, "w") as f_out:
            json.dump(combined_data, f_out, indent=4)

        print(f"Data successfully merged and saved to {output_file}")


    @staticmethod
    def add_number_to_json_entries(file_path):
        """
        Adds a "number" value to each JSON element, equal to its ordinal number in the array of values.
        Necessary for convenient work with the dataset.

        :param file_path: Path to the JSON file to process.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        for index, item in enumerate(data):
            item['number'] = index + 1

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Data successfully updated and saved to {file_path}.")


    @staticmethod
    def test_api(output_file):
        """
        Execute the API test and save the output to a JSON file.

        :param output_file: The name of the file to save the JSON data.
        """
        o = api.data()
        print(o)  # Optional: Print the DataFrame to the console

        # Convert the DataFrame to JSON
        json_output = o.to_json(orient="records", indent=4)

        # Save the JSON to a file
        with open(output_file, "w") as f:
            f.write(json_output)

        print(f"Output saved to {output_file}")