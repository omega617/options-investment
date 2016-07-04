from pandas import read_csv, Series
import pandas as pd
import glob
import bz2
import numpy as np
import re
import datetime
from numpy import nan
import math

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

if __name__ == '__main__':
    all_data = {}
    path =r'C:/Temp/algoseek/TRADES_ONLY/' # use your path
    allFilesZipped = glob.glob(path + "/*.csv.bz2")
    allFilesZipped.sort()
    frame = pd.DataFrame()
    list_ = []
    days_with_data = TradeDays()
    for zippedFile_ in allFilesZipped:
        if zippedFile_.__contains__("2007"):
            file_ = bz2.BZ2File(zippedFile_)
            df = pd.read_csv(file_,index_col=None, header=0)
            # extract the trade date from the zip file
            # re_result = re.search('(20\d{2})(\d{2})(\d{2})', zippedFile_)
            trade_date = re.search('(20\d{2})(\d{2})(\d{2})', zippedFile_)
            # add the extracted trade date as another column
            year = trade_date.group(1)
            month = trade_date.group(2)
            day = trade_date.group(3)
            df['Date'] = Series(np.repeat(pd.to_datetime(trade_date.group(0)), len(df)), index=df.index)
            df['Year'] = Series(np.repeat(year, len(df)), index=df.index)
            df['Month'] = Series(np.repeat(month, len(df)), index=df.index)
            df['Day'] = Series(np.repeat(day, len(df)), index=df.index)
            days_with_data.add_day(year, month, day)
            list_.append(df)
    frame = pd.concat(list_)

    # to select an indivdual date use the following:
    # frame.loc[frame.Date == pd.to_datetime('20070606')]
    columns = ['Trades', 'Min', 'Max', 'Empties']
    summary_stats = pd.DataFrame(columns=columns, index=pd.PeriodIndex(start='1/2007', end='12/2007', freq='M'))
    pd.np.empty((len(df), 0)).tolist()
    summary_stats.fillna({'Trades':0, 'Max':0, 'Empties':0}, inplace=True)

    day_dates = []
    for month in set(frame.Month):
        int_month = int(month)
        for day in days_with_data.get_days(2007, int_month):
            day_date = pd.Timestamp(datetime.datetime(2007, int_month, day))
            day_dates.append(day_date)
    day_columns = ['Trades', 'MinStrike', 'MaxStrike']
    day_stats = pd.DataFrame(columns=day_columns, index=pd.PeriodIndex(day_dates, freq='D'))
    day_stats.fillna(0, inplace=True)

    for month in set(frame.Month):
        date = pd.Period('2007-' + month)
        month_frame = frame.loc[frame.Month == month]
        summary_stats.set_value(date, 'Trades', month_frame.shape[0])
        int_month = int(month)
        for day in days_with_data.get_days(2007, int_month):
            day_date = pd.Period(datetime.datetime(2007, int_month, day), freq='D')
            day_frame = month_frame.loc[month_frame.Date == pd.to_timestamp(day_date)]

            day_stats.set_value(day_date, 'Trades', day_frame.shape[0])
            day_stats.set_value(day_date, 'MinStrike', day_frame.Strike.min())
            day_stats.set_value(day_date, 'MaxStrike', day_frame.Strike.max())

            if day_frame.shape[0] == 0:
                print "Empty Day : %s".format(str(day_date))
                if summary_stats.loc[date].Empties == 0:
                    summary_stats.set_value(date, 'Empties', 1)
                else:
                    empties = summary_stats.loc[date].Empties
                    summary_stats.set_value(date, 'Empties', empties+1)

            if math.isnan(summary_stats.loc[date].Min):
                summary_stats.loc[date].Min = day_frame.shape[0]
            else:
                summary_stats.loc[date].Min = min(day_frame.shape[0], summary_stats.loc[date].Min)

            # Max is set to zero by default
            summary_stats.loc[date].Max = max(day_frame.shape[0], summary_stats.loc[date].Max)

    print(summary_stats)
    print(day_stats)
    # load_data_all(all_data)