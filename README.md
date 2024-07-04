# Portfolio Optimizer App - README

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
    - [PortfolioOptimizer Class](#portfoliooptimizer-class)
    - [Streamlit Application](#streamlit-application)
6. [Features](#features)
7. [Running the Application](#running-the-application)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
The Portfolio Optimizer App is a web-based application built using Streamlit that allows users to optimize their stock portfolios. The application leverages Yahoo Finance to fetch historical stock data and the SciPy library for optimization. It calculates the efficient frontier, optimal portfolio, and various metrics like Sharpe Ratio, volatility, and returns. Additionally, the app provides visualizations and comparisons between user-defined portfolios and optimized portfolios.

## Requirements
- Python 3.7 or higher
- Streamlit
- yfinance
- pandas
- numpy
- scipy
- matplotlib

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/portfolio-optimizer.git
   cd portfolio-optimizer

2. Install the required packages in the requirements.txt file:
   
   pip install -r requirements.txt
   
4. To start the application run the following command:
   
   streamlit run app.py

