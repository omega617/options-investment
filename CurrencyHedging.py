import numpy as np
import pandas as pd
from sys import stdin
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import HoverTool, BoxSelectTool

# FXA
aud_amount = 10000
aud_usd_rate = 0.712907483

usd_amount = aud_amount * aud_usd_rate

price_range = np.arange(62, 80, 1)
# ask price data from 26/5/2016 for FXA call chain at strikes b/w 62 - 77
jul16_canned_prices = {62 : 10.6, 63 : 9.5, 64 : 8.4, 65 : 7.4, 66 : 6.4, 67 : 5.4, 68 : 4.4, 69 : 3.6,
                 70 : 2.8, 71 : 1.95, 72 : 1.35, 73 : 0.85, 74 : 0.6, 75 : 0.4, 76 : 0.3, 77 : 0.35,
                 78 : 0.35, 79 : 0.3, 80 : 0.3}
sep16_canned_prices = {62 : 10.7, 63 : 9.7, 64 : 8.7, 65 : 7.7, 66 : 6.8, 67 : 5.8, 68 : 5.0, 69 : 4.1,
                 70 : 3.3, 71 : 2.6, 72 : 1.95, 73 : 1.45, 74 : 1.05, 75 : 0.8, 76 : 0.55, 77 : 0.45,
                 78 : 0.35, 79 : 0.5, 80 : 0.45}
# updated 30/5 from http://www.cboe.com/delayedquote/quotetable.aspx
dec16_canned_prices = {62 : 10.6, 63 : 9.7, 64 : 8.7, 65 : 7.8, 66 : 6.9, 67 : 6.1, 68 : 5.2, 69 : 4.4,
                 70 : 3.7, 71 : 3.1, 72 : 2.55, 73 : 2.1, 74 : 1.65, 75 : 1.35, 76 : 1.05, 77 : 0.85,
                 78 : 0.7, 79 : 0.55, 80 : 0.45}

class OptionValues:
    def __init__(self):
        self.strikes = []

    def addProfit(self, strike, profit):
        self.strikes.append((strike, profit))

def prices_for_strikes():
    prices = {}

    # get prices from user
    for strike in price_range:
        print("Please enter call price for strike ", strike)
        x = stdin.readline()
        prices[strike] = float(x)

    return prices

#
def calculate_option_pl(strike, price, contract_size):
    gains = []
    for exchange_rate in price_range:
        diff = float(exchange_rate - strike)
        option_profit = max(0, (diff * contract_size) * 100/exchange_rate)
        premium_loss = (price * contract_size) * 100/exchange_rate
        currency_gain = (usd_amount * 100/exchange_rate) - aud_amount
        result = currency_gain + option_profit - premium_loss
        gains.append((exchange_rate, result))
    return gains

def graph_results(initial_exchange, contract_size, pnls):
    TOOLS = 'pan,wheel_zoom,box_select,crosshair,resize,reset,hover'
    p = figure(title="Contract size {0}, exchange {1}".format(contract_size, initial_exchange), width=750, height=450, x_axis_label='Exchange rate', y_axis_label='Profit', tools=TOOLS)
    colors = ['blue', 'brown', 'green', 'red', 'yellow', 'cyan', 'grey', 'orange', 'indigo',
              'violet', 'black', 'teal', 'tan', 'silver', 'navy', 'gold', 'purple', 'aqua']
    i = 0
    for strike in price_range:
        x, y = map(list, zip(*pnls[strike]))
        # add a line renderer with legend and line thickness
        p.line(x, y, legend="Strike : " + str(strike), line_color=colors[i], line_width=2)
        p.circle(x, y, legend="Strike : " + str(strike), fill_color=colors[i], line_color=colors[i])
        i += 1

    return p

def generate_scenarios(contract_sizes, prices):
    for contract_size in contract_sizes:
        pnls = {}
        price_range.sort()
        for strike in price_range:
            pnls[strike] = calculate_option_pl(strike, prices[strike], contract_size)

        p = graph_results(aud_usd_rate, contract_size, pnls)
        graphs.append(p)

    return graphs

if __name__ == '__main__':
    contract_sizes = [80, 100, 120, 140, 160]
    graphs = []
    # output to static HTML file
    # output_file("jul16_fax_options.html", title="FXA protection")
    # graphs = generate_scenarios(contract_sizes, jul16_canned_prices)
    # # put all the plots in a grid layout
    # p = gridplot([[graphs[0], graphs[1]], [graphs[2], graphs[3]], [graphs[4], None]])
    # show(p)

    output_file("sep16_fax_options.html", title="FXA protection")
    graphs = generate_scenarios(contract_sizes, sep16_canned_prices)
    # put all the plots in a grid layout
    p = gridplot([[graphs[0], graphs[1]], [graphs[2], graphs[3]], [graphs[4], None]])
    show(p)

    # output_file("dec16_fax_options.html", title="FXA protection")
    # graphs = generate_scenarios(contract_sizes, dec16_canned_prices)
    # # put all the plots in a grid layout
    # p = gridplot([[graphs[0], graphs[1]], [graphs[2], graphs[3]], [graphs[4], None]])
    # show(p)