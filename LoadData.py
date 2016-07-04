from pandas import read_csv, Series
import pandas as pd
import glob
import bz2
import numpy as np
import re
import datetime
from jinja2 import Environment, FileSystemLoader
import math
from weasyprint import HTML

# def transform_frame(df):
#     #

# hold a mapping from year-month to list of days with trading data
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

def load_options_data():
    path =r'C:/Temp/algoseek/TRADES_ONLY/' # use your path
    allOptionDataFilesZipped = glob.glob(path + "/*.csv.bz2")
    allOptionDataFilesZipped.sort()
    options_data_frame = pd.DataFrame()
    list_ = []
    days_with_options_data = TradeDays()
    for zippedOptionsDataFile_ in allOptionDataFilesZipped:
        if zippedOptionsDataFile_.__contains__("2007"):
            file_ = bz2.BZ2File(zippedOptionsDataFile_)
            df = pd.read_csv(file_,index_col=None, header=0)
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
    return pd.concat(list_), days_with_options_data

if __name__ == '__main__':
    spx_data_frame = load_spx_data()
    options_data_frame, days_with_options_data = load_options_data()

    # to select an individual date use the following:
    # frame.loc[frame.Date == pd.to_datetime('20070606')]
    columns = ['TotalTrades', 'MinTrades', 'MaxTrades', 'Empties']
    summary_stats = pd.DataFrame(columns=columns, index=pd.PeriodIndex(start='1/2007', end='12/2007', freq='M'))
    summary_stats.fillna({'TotalTrades':0, 'MaxTrades':0, 'Empties':0}, inplace=True)

    day_dates = []
    for month in sorted(set(options_data_frame.Month)):
        int_month = int(month)
        for day in days_with_options_data.get_days(2007, int_month):
            day_date = pd.Timestamp(datetime.datetime(2007, int_month, day))
            day_dates.append(day_date)
    day_columns = ['TotalTrades', 'MinStrike', 'MaxStrike', 'SPX']
    day_stats = pd.DataFrame(columns=day_columns, index=pd.PeriodIndex(day_dates, freq='D'))
    day_stats.fillna(0, inplace=True)

    for month in sorted(set(options_data_frame.Month)):
        date = pd.Period('2007-' + month)
        month_frame = options_data_frame.loc[options_data_frame.Month == month]
        summary_stats.set_value(date, 'TotalTrades', month_frame.shape[0])
        int_month = int(month)
        for day in days_with_options_data.get_days(2007, int_month):
            day_date = pd.Period(datetime.datetime(2007, int_month, day), freq='D')
            day_frame = month_frame.loc[month_frame.Date == day_date]

            day_stats.set_value(day_date, 'TotalTrades', day_frame.shape[0])
            day_stats.set_value(day_date, 'MinStrike', day_frame.Strike.min() / 1000)
            day_stats.set_value(day_date, 'MaxStrike', day_frame.Strike.max() / 1000)
            day_stats.set_value(day_date, 'SPX', spx_data_frame[str(day_date)]['SPX'][0])

            if day_frame.shape[0] == 0:
                print "Empty Day : %s".format(str(day_date))
                if summary_stats.loc[date].Empties == 0:
                    summary_stats.set_value(date, 'Empties', 1)
                else:
                    empties = summary_stats.loc[date].Empties
                    summary_stats.set_value(date, 'Empties', empties+1)

            if math.isnan(summary_stats.loc[date].MinTrades):
                summary_stats.set_value(date, 'MinTrades', day_frame.shape[0])
            else:
                summary_stats.set_value(date, 'MinTrades', min(day_frame.shape[0], summary_stats.loc[date].MinTrades))

            # Max is set to zero by default
            summary_stats.set_value(date, 'MaxTrades', max(day_frame.shape[0], summary_stats.loc[date].MaxTrades))

    summary_stats.sort_index(inplace=True)
    print(summary_stats)
    day_stats.sort_index(inplace=True)
    print(day_stats)
    print('Max. strike for 2007: {0}').format(day_stats['MaxStrike'].max().max())
    print('Min. strike for 2007: {0}').format(day_stats['MinStrike'].min().min())

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("myreport.html")
    template_vars = {"title" : "AlgoSeek Options Data Report",
                 "monthly_options_table": summary_stats.to_html(),
                 "daily_options_table": day_stats.to_html()}
    html_out = template.render(template_vars)
    HTML(string=html_out).write_pdf("report.pdf")
    # load_data_all(all_data)