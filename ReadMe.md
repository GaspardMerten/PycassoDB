# SNCB Predictive Maintenance

## Installation

Run the following command to install the project dependencies:

```shell
pip install -r requirements.txt
```

## Usage

### Running the Project

To run the project, execute the following command:

Do make sure that the `config.toml` file is present in the project root directory.

```shell
python main.py
```

### Configuration

Configure the project using the `config.toml` file. This file contains global settings and component-specific
configurations.

#### Global Settings

- `storage_folder`: Specifies the folder for data storage.
- `runner_persistence`: Path to the runner persistence file.

#### Component Configuration

Each component can be configured under the `[components]` section.

### Example

```toml
[globals]
storage_folder = "data_2"
runner_persistence = "data_2/runner_persistence.json"

[components]
[components.name]
multiple_outputs = true
```

## Creating New Components

1. Create a new Python file under the `components` directory.
2. Implement your component by inheriting from the base component class. Refer to existing components like `dbscan.py`
   for guidance.
3. Register your component in the config file under the `[components]` section.

## Project Structure

- `src/`: Contains the source code.
    - `framework/`: Core framework modules like `runner.py`, `config.py`, etc.
    - `mining/`: Modules for data mining techniques.
- `components/`: Custom components like `dbscan.py`, `kmeans.py`, etc.

