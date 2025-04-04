# Weather Prediction with AI and GitHub Actions

This repository combines the power of Artificial Intelligence (AI) and GitHub Actions to automatically gather, analyze, and predict weather data. The project is designed to be fully automated, leveraging GitHub Actions for continuous data collection and model training, all within a structured and organized workflow.

# Project Overview

## Description

The goal of this project is to automate the process of obtaining weather data, analyzing it, and making accurate weather forecasts using AI. The system is designed to collect weather data periodically, store it in a separate branch ([data](https://github.com/GitGinocchio/weather-prediction-with-github-actions/tree/data)), and use this data to train an AI model that predicts future weather conditions.

## Key Features

- **Automated Data Collection**: A GitHub Action periodically collects weather data from the [wttr.in](https://github.com/chubin/wttr.in) api and stores it in the `data` branch of this repository.
- **AI-based Weather Forecasting**: The collected data is used to train a machine learning model, which predicts future weather conditions.
- **Organized Data Management**: The `main` branch contains the core code, while the `data` branch stores all the weather data, keeping the repository clean and efficient.
- **Customizable Workflow**: The project is designed to be flexible and easily customizable to suit different weather APIs, data analysis methods, and machine learning models.

## How It Works

### GitHub Actions Workflow
1. **Data Collection**: The [`action.yml`](.github/workflows/collect_weather_data.yml) is triggered every one hour. It performs the following steps:

    - Checks out the main branch to run the Python script for collecting weather data.
    - Checks out the data branch to store the collected data.
    - Commits and pushes the updated database to the data branch.
2. **Data Analysis and Forecasting**: 
   
    `still in development...`
    <!-- An optional GitHub Action (examine-data.yml) can be set up to analyze the data and generate weather forecasts based on the collected data. -->
3. **Forecasting Website**: a static website for weather forecasts using GitHub Pages
   
    `still in development...`

## Repository Structure

### Branches

- **main**: Contains the core Python scripts and configuration files for data collection and model training.
- **data**: Stores all the weather data collected by the GitHub Actions. This branch is automatically updated with new data but is kept separate from the main codebase to maintain efficiency.

#### Main Branch
```python
├───.github/
│   └──workflows/
│       └───collect_weather_data.yml # GitHub Action for data collection and for data analyzing
├───config  # Where all configuration files are saved
├───models  # Folder where all the pre-trained models are present
│   ├───best-models
│   ├───sklearn-models
│   └───torch-models
├───src     # Folder where all the source code is present
│   ├───actions # Folder containing the scripts for the various GitHub Actions
│   └───models  # Folder containing scripts for creating separate models for each library
│       └───utils   # Folder containing a set of scripts used for general utilities
├───action-requirements.txt # Requirements to launch GitHub Actions
├───action.yml
└───README.md   # This file : )
```

#### Data Branch

```python
├───...
└───database.db # Database containing all the weather data collected by the GitHub Actions
```

## Setup Instructions

#### Prerequisites

- Python 3.9 or later

#### Installation

1. **Clone the repository:**

   ```bash
   git clone GitGinocchio/WeatherPredictionWithGithubActions.git
   cd WeatherPredictionWithGithubActions
   ```

2. **Install dependencies:**
    ```bash
    pip install -r action-requirements.txt
    pip install -r dev-requirements.txt
    ```
<!-- 
3. **Configure GitHub Actions:**

   The data collection workflow (```.github\workflows\actions.yml```) is already configured to run every one hours. 
   You can customize the schedule by modifying the cron expression.
   The default cron is: ```0 0/1 * * *``` see [Crons Explanation](https://crontab.guru/#0_0/2_*_*_*) for help
-->

#### Running Locally
    
to run the data collection script locally:
```bash
python src/collect_weather_data.py
```
#### Data Storage

All weather data is stored in the data branch of this repository. You can switch to this branch to view or download the data:
```bash
git checkout data
```

### Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.