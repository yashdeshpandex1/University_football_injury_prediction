# University Football Injury Prediction

⚽ **Predicting Football Player Injuries Using Machine Learning**

This project uses historical football player data to predict whether a player is likely to get injured in the next season. It includes a **Random Forest Classifier** model and a **Streamlit web app** for interactive predictions and visualization of feature importance.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Screenshots](#screenshots)
- [License](#license)

---

## Features

- Train a **Random Forest Classifier** on football player data.
- Input numeric and categorical player information to get a **real-time injury prediction**.
- Visualize **top features influencing the predictions**.
- Interactive **Streamlit app** with an intuitive sidebar for inputs.
- Custom **football background and semi-transparent charts** for better UX.

---

## Dataset

The project uses a CSV dataset (`data.csv`) containing player information. Example features include:

- `Age`
- `Weight`
- `Position`
- `Matches_Played`
- `Minutes_Played`
- `Injury_Next_Season` (target)

Make sure your dataset includes a **target column named `Injury_Next_Season`**.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/University_football_injury_prediction.git
cd University_football_injury_prediction
