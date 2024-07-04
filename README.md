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


## Usage

   To start the application, navigate to the project directory and run:

    streamlit run app.py
   
## Components
### PortfolioOptimizer Class

The PortfolioOptimizer class handles all the backend calculations and data fetching. It includes methods to download stock data,     calculate returns, optimize the portfolio, and generate various plots. Here's a brief overview of its methods:

- **__init__**: Initializes the class with stock list, date range, period, and number of portfolios.

- **download_data**: Downloads adjusted close prices for the given stock list.

- **get_stock_info**: Fetches detailed information about each stock.

- **download_benchmark_data**: Downloads NIFTY50 benchmark data.

- **benchmark_returns**: Calculates benchmark returns.

- **calculate_returns**: Calculates daily returns for the stocks.

- **risk_free_rate_of_return**: Fetches the 10-year US Treasury Bond rate as the risk-free rate.

- **portfolio_return**: Calculates the portfolio's expected return.

- **portfolio_volatility**: Calculates the portfolio's volatility.
    
- **negative_sharpe_ratio**: Calculates the negative Sharpe Ratio for optimization.
    
- **calculate_efficient_frontier**: Generates the efficient frontier for the given stocks.
    
- **tracking_error_user**: Calculates the tracking error for user-defined weights.
    
- **tracking_error_best_portfolio**: Calculates the tracking error for the optimized portfolio.
    
- **best_sharpe_ratio_portfolio**: Returns the portfolio with the highest Sharpe Ratio.
    
- **plot_graph**: Plots the efficient frontier.
    
- **print_all_portfolios**: Displays all generated portfolios with their metrics.
    
- **plot_weights_bar_chart**: Compares optimal and user-defined portfolio weights.
    
- **plot_weights_pie_chart**: Plots a pie chart of portfolio weights.
    
- **plot_portfolio_movement**: Compares the movement of the optimized portfolio, user portfolio, and benchmark over time.

### Streamlit Application
The Streamlit application provides a user-friendly interface to interact with the **PortfolioOptimizer** class. Users can select stocks, define portfolio weights, set date ranges, and view various visualizations and metrics.

#### Sidebar
- **Stock Selection**: Users can select the number of stocks and choose specific companies and exchanges.
- **Date Range**: Users can set the start and end dates for the analysis.
- **Period**: Users can choose between yearly and monthly periods.
- **User Portfolio Weights**: Users can define their own portfolio weights.
- **Frequency**: Users can select the frequency (daily, monthly, yearly) for visualizations.

#### Pages

- **Stock Information**: Displays the latest one-day stock data and detailed information about the selected stocks.
  
- **Stock Optimizer**: Optimizes the portfolio, displays the efficient frontier, and compares the user-defined portfolio with the optimized portfolio.

## Features

- Download historical stock data from Yahoo Finance
- Calculate daily and periodic returns
- Optimize portfolio based on Sharpe Ratio
- Generate and plot the efficient frontier
- Compare user-defined portfolio with optimized portfolio
- Display detailed stock information and financial metrics
- Visualize portfolio weights and cumulative returns

## Running the Application

To run the application, ensure that you have followed the installation steps. Then, navigate to the project directory and run:
       
    streamlit run app.py

## Project Structure

    portfolio-optimizer/
    ├── app.py
    ├── portfolio_optimizer.py
    ├── requirements.txt
    └── README.md

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


   


