# ⚡ IND320 - Energy Analysis Platform

## 🌟 Overview

Welcome to the **IND320 Energy Analysis Platform**! This repository contains all files related to the IND320 project, including the Streamlit application for interactive analysis and the underlying Jupyter Notebooks used for model development.

The Streamlit app is a multi-page application designed for **exploring energy data, performing time-series analysis, and visualizing forecasting results**.

## 📂 Repository Structure

The project is organized into logical folders to separate the application code from the development work:

| Folder           | Content                                | Purpose                                                                                                     |
| :--------------- | :------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **`streamlit/`** | `app.py`, `pages/`, `requirements.txt` | Contains the **Streamlit application code** for interactive display and navigation.                         |
| **`notebooks/`** | 4 Jupyter Notebooks                    | Used for initial **data exploration**, **statistical modeling**, and **model development** (e.g., SARIMAX). |
| **`utils/`**     | `Data_loader.py`, etc.                 | Utility files and helper functions for data handling.                                                       |
| **`data/`**      | (Optional) CSVs, raw files             | Storage for primary data sources, if not loaded via API.                                                    |

---

## 🚀 Local Setup and Execution

Follow these steps to set up the environment and run the app locally on a Windows machine (using PowerShell is assumed, but standard Command Prompt/Terminal works too).

### 1. Install Requirements

Navigate to the Streamlit directory and install all necessary Python packages:

```powershell
# Navigate to the Streamlit directory
$ cd streamlit/
# Install dependencies
$ pip install -r requirements.txt
Run the application

$ streamlit run streamlit_app.py


# IND320 - Streamlit App

IND320 Streamlit App 🚀
Welcome to the IND320 Streamlit App. This application collects a set of pages for exploring data, visualizations, and interactive analyses. Use the sidebar to open any page.

What to expect on each page:

📈 Data Table: Open this page to access tools, visualizations, tables, or analyses related to data table.
🎯 Elhub Data: Open this page to access tools, visualizations, tables, or analyses related to elhub data.
📊 Map & Snowdrift: Open this page to access tools, visualizations, tables, or analyses related to map & snowdrift.
🔬 Outliers: Open this page to access tools, visualizations, tables, or analyses related to outliers.
🔍 Production Analysis: Open this page to access tools, visualizations, tables, or analyses related to production analysis.
📈 Sarimax Forecast: Open this page to access tools, visualizations, tables, or analyses related to sarimax forecast.
🎯 Stl & Spectogram: Open this page to access tools, visualizations, tables, or analyses related to stl & spectogram.
📊 Weather Energy: Open this page to access tools, visualizations, tables, or analyses related to weather energy.

This README shows how to set up the environment and run the app locally on Windows (PowerShell). It also includes tips for common problems.
```
