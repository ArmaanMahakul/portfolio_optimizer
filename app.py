import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt
from datetime import datetime

class PortfolioOptimizer:
    def __init__(self, stock_list, start_date, end_date, period, num_ports=150):
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.risk_free_rate = self.risk_free_rate_of_return()
        self.num_ports = num_ports

        try:
            self.data = self.download_data()
            self.daily_returns = self.calculate_returns()
            self.mean_returns = self.daily_returns.mean() * self.period
            self.covariance_matrix = self.daily_returns.cov() * np.sqrt(self.period)
            self.all_weights = None
            self.ret_arr = None
            self.vol_arr = None
            self.sharpe_arr = None
            self.benchmark_data = self.download_benchmark_data()

            self.check_for_nans_infs()
        except Exception as e:
            st.error("Kindly check the dates and stock tickers properly. Kindly check Yahoo Finance website to find if a particular stock is on some exchange or not.")
            st.stop()


    def check_for_nans_infs(self):
        assert not np.isnan(self.data.values).any(), "Data contains NaNs"
        assert not np.isinf(self.data.values).any(), "Data contains infinite values"
        assert not np.isnan(self.daily_returns.values).any(), "Daily returns contain NaNs"
        assert not np.isinf(self.daily_returns.values).any(), "Daily returns contain infinite values"
        assert not np.isnan(self.mean_returns).any(), "Mean returns contain NaNs"
        assert not np.isinf(self.mean_returns).any(), "Mean returns contain infinite values"
        assert not np.isnan(self.covariance_matrix.values).any(), "Covariance matrix contains NaNs"
        assert not np.isinf(self.covariance_matrix.values).any(), "Covariance matrix contains infinite values"

    # Function to download stock data
    def download_data(self):
        data = yf.download(self.stock_list, start=self.start_date, end=self.end_date)
        data = data.dropna()
        return data['Adj Close']

    # Function to fetch stock information
    def get_stock_info(self):
        stock_info_dict = {}
        for stock in self.stock_list:
            ticker = yf.Ticker(stock)
            stock_info = ticker.info
            stock_info_dict[stock] = stock_info
        return stock_info_dict

    # Function to download NIFTY50 data
    def download_benchmark_data(self):
        benchmark = yf.download('^NSEI', start=self.start_date, end=self.end_date)
        benchmark = benchmark['Adj Close']
        benchmark = benchmark.dropna()
        return benchmark

    # Function to calculate benchmark returns
    def benchmark_returns(self):
        daily_return_benchmark = self.benchmark_data.pct_change().dropna().mean() * self.period
        return daily_return_benchmark

    # Function to calculate returns
    def calculate_returns(self):
        daily_returns = self.data.pct_change().dropna()
        return daily_returns

    # The risk free rate of return is the 10year US Treasury Bond rate
    def risk_free_rate_of_return(self):
        free_data = yf.download("^TNX", self.start_date, self.end_date)
        free_data = free_data["Adj Close"]
        free_data = free_data.dropna()
        return free_data[-1] / 100

    # Calculating portfolio returns
    def portfolio_return(self, weights):
        return np.sum(self.mean_returns * weights)

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights.transpose())))

    def negative_sharpe_ratio(self, weights):
        p_ret = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        return - (p_ret - self.risk_free_rate) / p_vol
    def calculate_efficient_frontier(self):
        num_assets = len(self.mean_returns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        w_min = np.ones(num_assets) / num_assets  # Initial guess for weights

        self.all_weights = np.zeros((self.num_ports, num_assets))
        self.ret_arr = np.zeros(self.num_ports)
        self.vol_arr = np.zeros(self.num_ports)
        self.sharpe_arr = np.zeros(self.num_ports)

        sum_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        min_var_result = minimize(self.portfolio_volatility, w_min, method='SLSQP', bounds=bounds,
                                  constraints=sum_constraint)
        w_min = min_var_result.x

        max_sharpe_result = minimize(self.negative_sharpe_ratio, w_min, method='SLSQP', bounds=bounds,
                                     constraints=sum_constraint)
        w_sharpe = max_sharpe_result.x

        self.ret_arr[0] = self.portfolio_return(w_min)
        self.ret_arr[1] = self.portfolio_return(w_sharpe)
        self.vol_arr[0] = self.portfolio_volatility(w_min)
        self.vol_arr[1] = self.portfolio_volatility(w_sharpe)
        self.all_weights[0, :] = w_min
        self.all_weights[1, :] = w_sharpe

        min_return = self.ret_arr[0]
        max_return = self.ret_arr[1]
        gap = (max_return - min_return) / (self.num_ports - 2)

        for i in range(2, self.num_ports):
            port_ret = min_return + (i) * gap
            double_constraint = LinearConstraint([np.ones(num_assets), self.mean_returns], [1, port_ret], [1, port_ret])
            x0 = w_min
            fun = lambda w: self.portfolio_volatility(w)
            a = minimize(fun, x0, method='trust-constr', constraints=double_constraint, bounds=bounds)
            self.all_weights[i, :] = a.x
            self.ret_arr[i] = port_ret
            self.vol_arr[i] = self.portfolio_volatility(a.x)

        self.sharpe_arr = (self.ret_arr - self.risk_free_rate) / self.vol_arr

        return self.vol_arr, self.ret_arr, self.sharpe_arr, self.all_weights

    def tracking_error_user(self, user_weights):
        user_portfolio_returns = np.dot(self.daily_returns, user_weights)
        benchmark = self.benchmark_returns()
        return np.sqrt(np.mean(((user_portfolio_returns - benchmark) ** 2)) / self.num_ports)

    def tracking_error_best_portfolio(self):
        max_sharpe_idx = self.sharpe_arr.argmax()
        max_sharpe_weights = self.all_weights[max_sharpe_idx]
        max_sharpe_portfolio_returns = np.dot(self.daily_returns, max_sharpe_weights)
        benchmark = self.benchmark_returns()
        return np.sqrt(np.mean((max_sharpe_portfolio_returns - benchmark) ** 2)/ self.num_ports)

    def best_sharpe_ratio_portfolio(self):
        max_sharpe_idx = self.sharpe_arr.argmax()
        max_sharpe_weights = self.all_weights[max_sharpe_idx]
        max_sharpe_return = self.ret_arr[max_sharpe_idx]
        max_sharpe_volatility = self.vol_arr[max_sharpe_idx]
        max_sharpe_ratio = self.sharpe_arr[max_sharpe_idx]
        return max_sharpe_weights, max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio

    def plot_graph(self):
        vol, ret, sharpe, all_weights = self.calculate_efficient_frontier()
        plt.figure(figsize=(10, 6))
        plt.scatter(vol, ret, c=sharpe, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        st.pyplot(plt)

    def print_all_portfolios(self):
        portfolios = []
        for i in range(self.num_ports):
            portfolio = {
                'Expected Return': self.ret_arr[i] * 100,
                'Volatility': self.vol_arr[i],
                'Sharpe Ratio': self.sharpe_arr[i]
            }
            for j, stock in enumerate(self.stock_list):
                portfolio[stock] = self.all_weights[i, j] * 100
            portfolios.append(portfolio)

        portfolios_df = pd.DataFrame(portfolios).sort_values(by='Sharpe Ratio', ascending=False)

        max_sharpe_idx = portfolios_df['Sharpe Ratio'].idxmax()

        def highlight_row(row):
            return ['background-color: yellow'] * len(row) if row.name == max_sharpe_idx else ['background-color: green'] * len(row)

        styled_df = portfolios_df.style.apply(highlight_row, axis=1)

        st.dataframe(styled_df)
    def plot_weights_bar_chart(self, optimal_weights, user_weights):
        fig, ax = plt.subplots()
        bar_width = 0.10
        index = np.arange(len(self.stock_list))

        ax.bar(index, optimal_weights * 100, bar_width, label='Optimal Portfolio')
        ax.bar(index + bar_width, user_weights * 100, bar_width, label='User Portfolio')

        ax.set_ylabel('Weight (%)')
        ax.set_title('Portfolio Weights Comparison')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(self.stock_list)
        ax.legend()

        st.pyplot(fig)

    def plot_weights_pie_chart(self, weights, title):
        fig, ax = plt.subplots()
        ax.pie(weights, labels=self.stock_list, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(title)
        st.pyplot(fig)

    def plot_portfolio_movement(self, optimal_weights, user_weights, freq='D'):
        optimal_portfolio = (self.daily_returns * optimal_weights).sum(axis=1)
        user_portfolio = (self.daily_returns * user_weights).sum(axis=1)
        benchmark = self.benchmark_data.pct_change().dropna()

        optimal_portfolio_cum = (1 + optimal_portfolio).cumprod()
        user_portfolio_cum = (1 + user_portfolio).cumprod()
        benchmark_cum = (1 + benchmark).cumprod()

        if freq == 'M':
            optimal_portfolio_cum = optimal_portfolio_cum.resample('M').last()
            user_portfolio_cum = user_portfolio_cum.resample('M').last()
            benchmark_cum = benchmark_cum.resample('M').last()
        elif freq == 'Y':
            optimal_portfolio_cum = optimal_portfolio_cum.resample('Y').last()
            user_portfolio_cum = user_portfolio_cum.resample('Y').last()
            benchmark_cum = benchmark_cum.resample('Y').last()

        fig, ax = plt.subplots()
        ax.plot(optimal_portfolio_cum, label='Optimal Portfolio')
        ax.plot(user_portfolio_cum, label='User Portfolio')
        ax.plot(benchmark_cum, label='Benchmark (^NSEI)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Portfolio and Benchmark Movement')
        ax.legend()
        st.pyplot(fig)

def main():
    st.sidebar.header("Portfolio Optimizer App")
    page = st.sidebar.radio("Select a Page", ["Stock Information", "Stock Optimizer"])

    stock_symbols_df = pd.read_csv('Stock_Symbols.csv')
    companies = dict(zip(stock_symbols_df['Company_Name'], stock_symbols_df['Scrip']))

    st.sidebar.title("Stock Selection")
    num_stocks = st.sidebar.slider('Number of stocks', min_value=1, max_value=10, value=5)
    stock_list = []

    for i in range(num_stocks):
        company_name = st.sidebar.selectbox(f"Select Company {i + 1}", options=list(companies.keys()),
                                            index=i + 100)
        exchange = st.sidebar.selectbox(f"Select Exchange {i + 1}", ['NSE', 'BSE'])
        stock_symbol = companies[company_name]
        ticker = f"{stock_symbol}.BO" if exchange == "BSE" else f"{stock_symbol}.NS"
        stock_list.append(ticker)

    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    current_date = datetime.now().date()
    end_date = st.sidebar.date_input('End Date', value=current_date, max_value=current_date)
    period_options = {'Yearly': 252, 'Monthly': 30}
    period = st.sidebar.selectbox('Period', options=list(period_options.keys()))
    period_value = period_options[period]

    st.sidebar.title("User Portfolio Weights")
    user_weights = []
    for stock in stock_list:
        weight = st.sidebar.number_input(f'Weight for {stock}', value=100 / len(stock_list))
        user_weights.append(weight / 100)

    if sum(user_weights) != 1:
        st.sidebar.warning('The sum of weights must be 100')
    else:
        user_weights = np.array(user_weights)

        frequency = st.sidebar.selectbox("Select Frequency (For Visualization)", ["Daily", "Monthly", "Yearly"],
                                         index=0)
        freq_map = {"Daily": 'D', "Monthly": 'M', "Yearly": 'Y'}

        if page == 'Stock Optimizer':
            if st.sidebar.button('Optimize'):
                optimizer = PortfolioOptimizer(stock_list, str(start_date), str(end_date), period_value)
                st.header("Efficient Frontier Curve")
                optimizer.plot_graph()

                # Display optimal portfolio metrics
                max_sharpe_weights, max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio = optimizer.best_sharpe_ratio_portfolio()
                tracking_error = optimizer.tracking_error_user(user_weights)
                st.write("## Optimal Portfolio (Highest Sharpe Ratio)")
                st.write(f" Expected Return:    {max_sharpe_return * 100:.2f}%")
                st.write(f" Volatility:         {max_sharpe_volatility:.2f}")
                st.write(f" Sharpe Ratio:       {max_sharpe_ratio:.2f}")
                st.write(f"Tracking Error:      {tracking_error:.2f}")

                optimal_weights_df = pd.DataFrame({
                    'Stock': stock_list,
                    'Optimal Weights': max_sharpe_weights * 100,
                    'User Weights': user_weights * 100
                })
                st.write("### Optimal Portfolio Weights")
                st.table(optimal_weights_df)

                # Bar chart for optimal and user portfolio weights comparison
                optimizer.plot_weights_bar_chart(max_sharpe_weights, user_weights)

                # Calculate and display user portfolio metrics

                st.write("## User Portfolio Metrics")

                user_portfolio_return = optimizer.portfolio_return(user_weights)
                user_portfolio_volatility = optimizer.portfolio_volatility(user_weights)
                user_portfolio_sharpe_ratio = (
                                                      user_portfolio_return - optimizer.risk_free_rate) / user_portfolio_volatility
                tracking_error_portfolio = optimizer.tracking_error_best_portfolio()

                st.write("### User Portfolio Comparison")
                comparison_data = {
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Tracking Error'],
                    'Optimal Portfolio': [f"{max_sharpe_return * 100:.2f}%", f"{max_sharpe_volatility:.2f}",
                                          f"{max_sharpe_ratio:.2f}", f'{tracking_error_portfolio:.2f}'],
                    'User Portfolio': [f"{user_portfolio_return * 100:.2f}%", f"{user_portfolio_volatility:.2f}",
                                       f"{user_portfolio_sharpe_ratio:.2f}", f'{tracking_error:.2f}']
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)

                # Pie chart for optimal portfolio weights
                optimizer.plot_weights_pie_chart(max_sharpe_weights, "Optimal Portfolio Composition")

                # Pie chart for user portfolio weights
                optimizer.plot_weights_pie_chart(user_weights, "User Portfolio Composition")

                st.title("User Portfolio vs Max Sharpe Portfolio in line with Benchmark")
                optimizer.plot_portfolio_movement(max_sharpe_weights, user_weights, freq_map[frequency])
                optimizer.print_all_portfolios()

        else:
            load = PortfolioOptimizer(stock_list, str(start_date), str(end_date), period_value)

            # Fetch the latest one-day data for the selected stocks
            latest_data = {}
            for ticker in stock_list:
                data = yf.download(ticker, period='1d')
                if not data.empty:
                    latest_data[ticker] = data.iloc[-1]

            # Convert the latest data to a DataFrame for display
            if latest_data:
                latest_data_df = pd.DataFrame(latest_data).T
                latest_data_df.index.name = 'Ticker'
                st.subheader('Latest One-Day Stock Data')
                st.dataframe(latest_data_df)
            else:
                st.warning('No data available for the selected stocks.')

            data_plot = {}
            for tickers in stock_list:
                data = yf.download(tickers, start=start_date, end=end_date)
                data_plot[tickers] = data["Adj Close"]
                data_plot = pd.DataFrame(data_plot)

            fig, ax = plt.subplots()

            ax.plot(data_plot)
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.set_title('Stock Price vs Time')
            st.pyplot(fig)

            stock_info_dict = load.get_stock_info()

            for ticker, stock_info in stock_info_dict.items():
                st.subheader(f"Stock Information for {ticker}")

                # Displaying basic information
                st.write("### Basic Information")
                st.write(f"**Name**: {stock_info.get('shortName', 'N/A')}")
                st.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry**: {stock_info.get('industry', 'N/A')}")
                st.write(f"**Website**: [{stock_info.get('website', 'N/A')}]({stock_info.get('website', 'N/A')})")

                # Displaying financial information
                st.write("### Financials")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Market Cap**: {stock_info.get('marketCap', 'N/A')}")
                    st.write(f"**PE Ratio**: {stock_info.get('trailingPE', 'N/A')}")
                    st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A')}")
                    st.write(f"**EPS**: {stock_info.get('trailingEps', 'N/A')}")
                with col2:
                    st.write(f"**Revenue**: {stock_info.get('totalRevenue', 'N/A')}")
                    st.write(f"**Gross Profit**: {stock_info.get('grossProfits', 'N/A')}")
                    st.write(f"**Operating Margin**: {stock_info.get('operatingMargins', 'N/A')}")
                    st.write(f"**Profit Margin**: {stock_info.get('profitMargins', 'N/A')}")


if __name__ == '__main__':
    main()