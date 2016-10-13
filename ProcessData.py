from pandas import read_csv, Series
import pandas as pd
import glob
import bz2
import numpy as np
import re
import datetime
from jinja2 import Environment, FileSystemLoader
import math
from itertools import tee
import calendar
from pandas.tseries.offsets import BDay

contract_size = 100
years_for_analysis = [2012]#, 2013, 2014, 2015, 2016]
# option_buying_style = 0 : buy only the trades that we saw
# option_buying_style = 1 : use the premium for the trade we saw and assume we can get the quantity we want
option_buying_style = 1

RISKFREE = 0.035
VOLATILITY = 0.40

def cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)

def black_scholes(callResult, putResult, stockPrice, optionStrike, optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)

    expRT = np.exp(- R * T)
    callResult[:] = (S * cndd1 - X * expRT * cndd2)

def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


class TradeDays:
    def __init__(self, ):
        self.yearmonth_day = {}

    def add_day(self, year, month, day):
        key = int(year)*100 + int(month)
        if key in self.yearmonth_day:
            self.yearmonth_day[key].append(int(day))
        else:
            self.yearmonth_day[key] = [int(day)]

    def get_days(self, year, month):
        try:
            result = self.yearmonth_day[int(year)*100 + int(month)]
        except:
            result = []
        return result

def load_data(filename):
    df = read_csv("c:/Temp/algoseek/", filename)

def load_data_all(all_data):
    start_date = 20070103
    end_date = 20080103

    for date in xrange(start_date, end_date, step=1):
        all_data[date] = load_data(date)

def load_spx_data():
    path =r'C:/Temp/algoseek/' # use your path
    filename = 'dailypricehistory.xls'
    df = pd.read_excel(path+filename, index_col=0, header=0)
    return df

def load_options_data(month, year_for_analysis):
    date_for_data = str(year_for_analysis) + str(month)
    path =r'C:/Temp/algoseek/TRADES_ONLY/' # use your path
    allOptionDataFilesZipped = glob.glob(path + "/*.csv.bz2")
    allOptionDataFilesZipped.sort()
    options_data_frame = pd.DataFrame()
    list_ = []
    days_with_options_data = TradeDays()
    print('Loading data for {0}'.format(date_for_data))
    for zippedOptionsDataFile_ in allOptionDataFilesZipped:
        if zippedOptionsDataFile_.__contains__(str(date_for_data)):
            file_ = bz2.BZ2File(zippedOptionsDataFile_)
            df = pd.read_csv(file_, index_col=None, header=0, parse_dates=['Expiration'])
            # extract the trade date from the zip file
            # re_result = re.search('(20\d{2})(\d{2})(\d{2})', zippedFile_)
            options_trade_date = re.search('(20\d{2})(\d{2})(\d{2})', zippedOptionsDataFile_)
            # add the extracted trade date as another column
            year = options_trade_date.group(1)
            month = options_trade_date.group(2)
            day = options_trade_date.group(3)
            df['Date'] = Series(np.repeat(pd.Period(options_trade_date.group(0)), len(df)), index=df.index)
            df['Year'] = Series(np.repeat(year, len(df)), index=df.index)
            df['Month'] = Series(np.repeat(month, len(df)), index=df.index)
            df['Day'] = Series(np.repeat(day, len(df)), index=df.index)
            days_with_options_data.add_day(year, month, day)
            list_.append(df)
    if list_:
        options_data_frame = pd.concat(list_)
        options_data_frame = options_data_frame.set_index(['Date'])
    else:
        options_data_frame = pd.DataFrame()
    return options_data_frame, days_with_options_data



def process_day(bankroll, year_for_analysis, int_month, today, expiration_to_buy, month_frame, summary_stats):
    day_date = pd.Period(datetime.datetime(year_for_analysis, int_month, today), freq='D')

    import calendar

    c = calendar.Calendar(firstweekday=calendar.SUNDAY)

    day_frame = month_frame.loc[[day_date]]
    spx_day = spx_data_frame.ix[datetime.datetime(year_for_analysis,int_month,today,0,0)]['SPX'] / 10
    spx_minus_30 = spx_day - (0.3 * spx_day)

    # create criterion to select only options with 'expiration_to_buy'
    expiration_criterion = day_frame['Expiration'].map(lambda x: x.date() == expiration_to_buy.date())
    # create criterion to select only put options
    put_criterion = day_frame['PutCall'].map(lambda x: x == 'P')

    # select only those options from the day's trade data
    options_with_expiration = day_frame[expiration_criterion & put_criterion]

    # sort the options by how close they are to the strike we want (SPX - 30%)
    option = options_with_expiration.ix[((options_with_expiration.Strike / 10000) - spx_minus_30).abs().argsort()[:]]

    # sort options by the premium so we buy the cheapest ones first
    # option.sort_values(by='Premium', inplace=True)

    cheapest_option_index = 0
    list_ = []
    while bankroll > 0.0:
        contract = option.ix[cheapest_option_index]
        contracts_purchased = 0
        cost = 0.0

        # buy only the options that
        if option_buying_style == 0:
            while (cost < bankroll) & ((contract.Quantity - contracts_purchased) >= 1):
                contracts_purchased += 1
                cost += (contract.Premium / 10000.0) * contract_size

            contract.Quantity = contracts_purchased
            contract.Strike = contract.Strike / 10000
            contract['SPX'] = spx_day
            contract['SPX30'] = spx_minus_30
            list_.append(contract)
            bankroll = bankroll - cost
            cheapest_option_index += 1
        elif option_buying_style == 1:
            contract.Quantity = math.ceil(bankroll / ((contract.Premium / 10000.0) * contract_size))
            cost = (contract.Premium / 10000.0) * contract_size * contract.Quantity
            contract.Strike = (contract.Strike / 10000.0)
            contract['SPX'] = spx_day
            contract['SPX30'] = spx_minus_30
            list_.append(contract)
            bankroll = bankroll - cost
            cheapest_option_index += 1
    if list_:
        positions = pd.concat(list_, axis=1, keys=[s.name for s in list_])

    # print positions

    return positions, bankroll

# assumptions:
# - we can buy as many open_position contracts as we want at the given premium
# - we can sell all the contracts we buy at the historical premium
#
def sell_positions(bankroll, open_positions, today, int_month, year_for_analysis, month_frame):
    day_date = pd.Period(datetime.datetime(year_for_analysis, int_month, today), freq='D')

    day_frame = month_frame.loc[[day_date]]

    expiration_to_sell = open_positions.ix[0].Expiration

    # create criterion to select only options with 'expiration_to_sell'
    expiration_criterion = day_frame['Expiration'].map(lambda x: x.date() == expiration_to_sell.date())

    # create criterion to select only put options
    put_criterion = day_frame['PutCall'].map(lambda x: x == 'P')

    # select only those options from the day's trade data
    options_with_expiration = day_frame[expiration_criterion & put_criterion]

    open_positions_cost = 0.0
    for index, open_position in open_positions.iterrows():
        position_to_sell_strike = open_position.Strike
        # sort the options by how close they are to the strike of our open position
        closest_sold_option = options_with_expiration.ix[((options_with_expiration.Strike / 10000) - position_to_sell_strike).abs().argsort()[:]]
        spx_today = spx_data_frame.ix[datetime.datetime(year_for_analysis, int_month, today, 0, 0)]['SPX'] / 10

        i = 0
        # % of portfolio to invest into options for the tail hedge
        hedge_percent = 0.005
        while open_position.Quantity > 0:
            amount_to_buy = bankroll * hedge_percent
            contract_cost = open_position.Premium / 10000.0 * contract_size
            # we can only buy whole numbers of contracts, so use math.ceil
            open_position_contract_size = math.floor(amount_to_buy / contract_cost)
            open_positions_cost += open_position_contract_size * contract_cost
            # float(open_position.Quantity * (open_position.Premium / (bankroll * hedge_percent)) * contract_size)
            matched_strike = closest_sold_option.ix[i].Strike / 10000.0
            # get a premium that is as close as possible to that we should get paid
            matched_strike_premium = (float(closest_sold_option.ix[i].Premium) / 10000.0)
            if (matched_strike == position_to_sell_strike):
                premium = matched_strike_premium
            else:
                premium = matched_strike_premium * (float(position_to_sell_strike) / float(matched_strike))

            #print('Selling open position ({4} contracts) with strike {0} for {1} (premium for strike {2}, {3})'.format(position_to_sell_strike, premium, matched_strike, (float(closest_sold_option.ix[i].Premium) / 10000.0), open_position.Quantity))
            # loop over open positions, selling until we've sold all our equivalent positions
            #while (closest_sold_option.ix[i].Quantity > 0) and (open_position.Quantity > 0):
            profit_from_sell = premium * contract_size * open_position_contract_size
            add_to_bankroll = profit_from_sell - open_positions_cost
            open_position.Quantity = 0#open_position.Quantity - 1
            #closest_sold_option.ix[i].Quantity = closest_sold_option.ix[i].Quantity - 1
            i += 1
    print('Bought open positions (Strike={4}, Premium={5}, SPX={6}) for {0} sold for {1} (profit = {8}) (Strike={2}, Premium={3}, SPX={7})'.format(
        open_positions_cost,
        profit_from_sell,
        matched_strike,
        premium,
        position_to_sell_strike,
        open_position.Premium/10000.0,
        open_position.SPX,
        spx_today,
        add_to_bankroll
        )
    )
    if spx_today < open_position.SPX:
        print('SPX has dropped')
    spx_invest = bankroll * (1 - hedge_percent)
    spx_profit =  spx_invest * ((spx_today - open_position.SPX) / open_position.SPX)
    print('SPX @ open {0}, @ close {1}, invested : {2}, profit : {3}'.format(
        open_position.SPX,
        spx_today,
        spx_invest,
        spx_profit
        )
    )
    return add_to_bankroll, spx_profit

# buy option today if:
# exactly n (e.g. 90) days prior to expiration OR
# n-1 (e.g. 89) or n+1 (e.g. 91) days until expiration and
def nearestDate(today, tomorrow, dates, offset, buy_dates):
    nearness_today = { abs(today + datetime.timedelta(offset) - date) : date for date in dates }
    nearestDate_today = nearness_today[min(nearness_today.keys())]
    gap_between_today_and_nearest_expiration = abs(today - nearestDate_today)

    nearness_tomorrow = { abs(tomorrow + datetime.timedelta(offset) - date) : date for date in dates }
    nearestDate_tomorrow = nearness_tomorrow[min(nearness_tomorrow.keys())]
    gap_between_tomorrow_and_nearest_expiration = abs(tomorrow - nearestDate_tomorrow)

    offset_as_timedelta = datetime.timedelta(offset)
    # exact match between today and expiration
    if (gap_between_today_and_nearest_expiration == offset_as_timedelta):
        print 'Buy today! day_date : {0}, expiration : {1}, gap : {2} days'.format(today.date(), nearestDate_today.date(), gap_between_today_and_nearest_expiration.days)
        buy_dates[nearestDate_today] = True
        return nearestDate_today
    # exact match between tomorrow and expiration
    # elif gap_between_tomorrow_and_nearest_expiration == offset_as_timedelta:
    #     print 'Wait until tomorrow. day_date : {0}, expiration : {1}, gap : {2} days'.format(tomorrow.date(), nearestDate_tomorrow.date(), gap_between_tomorrow_and_nearest_expiration.days)
    #     return True
    # today is Monday and gap is (n-1) or (n-2)
    elif (gap_between_today_and_nearest_expiration == (offset_as_timedelta - datetime.timedelta(1))) & (today.weekday() == calendar.MONDAY):
        print 'Buy today! day_date : {0}, expiration : {1}, gap : {2} days'.format(today.date(), nearestDate_today.date(), gap_between_today_and_nearest_expiration.days)
        buy_dates[nearestDate_today] = True
        return nearestDate_today
    elif (gap_between_today_and_nearest_expiration == (offset_as_timedelta - datetime.timedelta(2))) & (today.weekday() == calendar.MONDAY):
        print 'Buy today! day_date : {0}, expiration : {1}, gap : {2} days'.format(today.date(), nearestDate_today.date(), gap_between_today_and_nearest_expiration.days)
        buy_dates[nearestDate_today] = True
        return nearestDate_today
    elif (gap_between_today_and_nearest_expiration < offset_as_timedelta) & (not nearestDate_today in buy_dates):
        print 'Buy today! day_date : {0}, expiration : {1}, gap : {2} days'.format(today.date(), nearestDate_today.date(), gap_between_today_and_nearest_expiration.days)
        buy_dates[nearestDate_today] = True
        return nearestDate_today

    return None
    # alternative : today is Friday and gap is (n+1)
    # elif (gap_between_today_and_nearest_expiration == (offset_as_timedelta - datetime.timedelta(2))) & (today.weekday() == calendar.TUESDAY) & monday was a holiday:


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

if __name__ == '__main__':
    spx_data_frame = load_spx_data()
    running_pnl = 0.0
    buy_dates = {}
    for year_for_analysis in years_for_analysis:
        # to select an individual date use the following:
        # frame.loc[frame.Date == pd.to_datetime('20070606')]
        columns = ['TotalTrades', 'MinTrades', 'MaxTrades', 'Empties', 'Under50', 'Under100', 'Under500', 'Under1000', 'Over1000']
        summary_stats = pd.DataFrame(columns=columns, index=pd.PeriodIndex(start='1/{0}'.format(year_for_analysis), end='12/{0}'.format(year_for_analysis), freq='M'))
        summary_stats.fillna({'TotalTrades':0, 'MaxTrades':0, 'Empties':0, 'Under50':0, 'Under100':0, 'Under500':0, 'Under1000':0, 'Over1000':0}, inplace=True)

        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        #months = ['04', '05', '06', '07', '08', '09', '10', '11', '12']
        total_bankroll = 200000
        bankroll = total_bankroll * 0.005
        open_positions = pd.DataFrame()
        running_pnl_options = 0
        initial_spx = 0
        final_spx = 0
        for month in sorted(set(months)):
            date = pd.Period(str(year_for_analysis)+'-' + month)
            #month_frame = options_data_frame.loc[options_data_frame.Month == month]
            month_frame, days_with_options_data = load_options_data(month, year_for_analysis)

            if month_frame.empty:
                break

            # get all the expiries traded in the month
            expirations = sorted(list(set(month_frame.Expiration)))

            int_month = int(month)
            day_dates = []
            for today, tomorrow in pairwise(days_with_options_data.get_days(year_for_analysis, int_month)):
                today_date = pd.Timestamp(datetime.datetime(year_for_analysis, int_month, today))
                tomorrow_date = pd.Timestamp(datetime.datetime(year_for_analysis, int_month, tomorrow))
                expiration_to_sell = nearestDate(today_date, tomorrow_date, expirations, 90, buy_dates)

                if not open_positions.empty:
                    open_positions = open_positions.T
                    for open_position_expiration in set(open_positions.Expiration):

                        gap = abs(open_position_expiration - today_date)
                        if gap <= datetime.timedelta(60):
                            print("Sell today! day_date : {0}, expiration {1}, gap : {2} days".format(today_date.date(), open_position_expiration.date(), gap.days))
                            open_position_expiration_criterion = open_positions.Expiration.map(lambda x: x.date() == open_position_expiration.date())
                            positions_to_sell = open_positions[open_position_expiration_criterion]

                            # get initial SPX to calculate overall profit for the year
                            if initial_spx == 0:
                                initial_spx = positions_to_sell.ix[0].SPX
                            final_spx = positions_to_sell.ix[0].SPX

                            options_pnl, spx_pnl = sell_positions(total_bankroll, positions_to_sell, today, int_month, year_for_analysis, month_frame)
                            open_positions = open_positions[open_position_expiration_criterion != True]
                            pnl = spx_pnl + options_pnl
                            running_pnl += pnl
                            running_pnl_options += options_pnl
                            print("PNL : {0}".format(pnl))
                            print("Running PNL : {0}".format(running_pnl))
                            print("Running PNL options : {0}".format(running_pnl_options))
                    open_positions = open_positions.T

                if expiration_to_sell is not None:
                    new_open_positions, bankroll = process_day((total_bankroll * 0.5), year_for_analysis, int_month, today, expiration_to_sell, month_frame, summary_stats)
                    running_pnl += bankroll
                    if open_positions.empty:
                        open_positions = new_open_positions
                    else:
                        open_positions = open_positions.join(new_open_positions)

        print("Running PNL options : {0}".format(running_pnl_options))
        spx_invest = (1 - 0.005) * total_bankroll
        spx_profit = spx_invest * ((final_spx - initial_spx) / initial_spx)
        print('SPX @ open {0}, @ close {1}, invested : {2}, profit : {3}'.format(
            initial_spx,
            final_spx,
            spx_invest,
            spx_profit
            )
        )
                # work out buy
        #     day_columns = ['TotalTrades', 'MinStrike', 'MaxStrike', 'SPX']
        #     day_stats_ = pd.DataFrame(columns=day_columns, index=pd.PeriodIndex(day_dates, freq='D'))
        #     day_stats_.fillna(0, inplace=True)
        #
        #     if month_frame is not None:
        #         summary_stats.set_value(date, 'TotalTrades', month_frame.shape[0])
        #         for day in days_with_options_data.get_days(year_for_analysis, int_month):
        #             process_day(int_month, day, month_frame, day_stats_, summary_stats)
        #
        #     list_.append(day_stats_)
        # day_stats = pd.concat(list_)
        #
        #
        # summary_stats.sort_index(inplace=True)
        # summary_stats.append(summary_stats.sum(numeric_only=True), ignore_index=True)
        # print(summary_stats)
        # day_stats.sort_index(inplace=True)
        # # print(day_stats)
        # print('Max. strike for {1}: {0}').format(day_stats['MaxStrike'].max().max(), year_for_analysis)
        # print('Min. strike for {1}: {0}').format(day_stats['MinStrike'].min().min(), year_for_analysis)
        # print('Min. SPX for {1}: {0}').format(day_stats['SPX'].min().min(), year_for_analysis)
        #
        # env = Environment(loader=FileSystemLoader('.'))
        # template = env.get_template("myreport.html")
        # template_vars = {"title" : "AlgoSeek Options Data Report {0}".format(year_for_analysis),
        #              "monthly_options_table": summary_stats.to_html(),
        #              "daily_options_table": day_stats.to_html()}
        # html_out = template.render(template_vars)
        # with open("options_data_{0}.html".format(year_for_analysis), "wb") as fh:
        #     fh.write(html_out)
        #
        # # load_data_all(all_data)