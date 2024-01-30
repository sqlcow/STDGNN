import pandas as pd
import numpy as np
import calendar


class prepareData2D:
    def __init__(self, graph_signal_matrix_filename_2D, num_of_hours, num_of_days, num_of_weeks, num_of_months,
                 num_of_years, end_of_months, start_date, end_date, linear_interpolation):
        self.graph_signal_matrix_filename_2D = graph_signal_matrix_filename_2D
        self.num_of_hours = int(num_of_hours)
        self.num_of_days = int(num_of_days)
        self.num_of_weeks = int(num_of_weeks)
        self.num_of_months = int(num_of_months)
        self.num_of_years = int(num_of_years)
        self.end_of_months = end_of_months
        self.start_date = start_date
        self.end_date = end_date
        self.linear_interpolation = linear_interpolation

    @staticmethod
    def check_column_names(df):
        # Check if the number of columns is at least less than 2
        if len(df.columns) < 2:
            raise Exception('Not enough columns')

        # Check if the first column is 'datetime' and the second column is 'id'
        if df.columns[0] == 'date_time' and df.columns[1] == 'id':
            return True
        elif df.columns[0] == 'id' and df.columns[1] == 'date_time':
            return True
        else:
            raise Exception(
                'The column name is wrong. The names of the first two columns of the 2D object need to be id and date_time.')

    @staticmethod
    def generate_last_days(start_year, end_year):
        last_days = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                _, last_day = calendar.monthrange(year, month)
                last_days.append(f"{year}-{month:02d}-{last_day:02d}")
        return last_days

    @property
    def prepareData2D(self):
        df = pd.read_csv(self.graph_signal_matrix_filename_2D)
        # Check whether the df object is legal
        self.check_column_names(df)
        if self.end_of_months == 'True':
            # generate the last day list
            start_year = int(self.start_date.split('-')[0])
            end_year = int(self.end_date.split('-')[0])
            last_days = self.generate_last_days(start_year, end_year)
            # Recode the id for merge
            df['Recode'] = df.groupby('id').ngroup()
            # generate the new dataframe which contains all the date_time and id
            date_time_list = last_days * (df['Recode'].max() + 1)
            df_new = pd.DataFrame({'date_time': date_time_list})
            df_new['Recode'] = np.repeat(np.arange(0, (df['Recode'].max() + 1)), len(last_days))
            # merge the new dataframe with the old dataframe
            df['date_time'] = pd.to_datetime(df['date_time'])
            df_new['date_time'] = pd.to_datetime(df_new['date_time'])
            df_new = pd.merge(df_new, df, on=['Recode', 'date_time'], how='left')
            df_new = df_new.drop(['Recode', 'date_time', 'id'], axis=1)
            # reshape the data and linear interpolation
            df_new = df_new.values.reshape(len(last_days), (df['Recode'].max() + 1), df_new.shape[1])
            print(f"after prepareData2D, the shape of the data is {df_new.shape}")
            # print(df_new[:, 0, 0].shape)
            # linear_interpolation
            if self.linear_interpolation == 'True':
                for k in range(df_new.shape[2]):
                    for i in range(df_new.shape[1]):
                        df_new[:, i, k] = pd.DataFrame(df_new[:, i, k]).interpolate().values.reshape(len(last_days))
                print("interpolate done")
                return df_new
            else:
                return df_new
        else:
            earliest_date = pd.to_datetime(df['date_time']).min()
            latest_date = pd.to_datetime(df['date_time']).max()
            print(
                f"Automatically identify the start and end date pattern: earliest date is {earliest_date} latest date is {latest_date}")
            if earliest_date == latest_date:
                raise Exception("The dataset contains only one time step. Please check the dataset.")
            if (self.num_of_hours + self.num_of_days + self.num_of_weeks + self.num_of_months + self.num_of_years) == 0:
                raise Exception("All the time steps are 0. Please check the dataset.")
            delta = pd.DateOffset(hours=self.num_of_hours, days=self.num_of_days, weeks=self.num_of_weeks,
                                  months=self.num_of_months, years=self.num_of_years)
            date_range = pd.date_range(earliest_date, latest_date, freq=delta)
            df['Recode'] = df.groupby('id').ngroup()
            date_time_list = date_range.tolist() * df['Recode'].max()
            df_new = pd.DataFrame({'date_time': date_time_list})
            df_new['Recode'] = np.repeat(np.arange(0, df['Recode'].max()), len(date_range))
            df['date_time'] = pd.to_datetime(df['date_time'])
            df_new['date_time'] = pd.to_datetime(df_new['date_time'])
            df_new = pd.merge(df_new, df, on=['Recode', 'date_time'], how='left')
            df_new = df_new.drop(['Recode', 'date_time', 'id'], axis=1)
            df_new = df_new.values.reshape(len(date_range), df['Recode'].max(), df_new.shape[1])
            print(f"after prepareData2D, the shape of the data is {df_new.shape}")
            # linear_interpolation
            if self.linear_interpolation == 'True':
                for k in range(df_new.shape[2]):
                    for i in range(df_new.shape[1]):
                        df_new[:, i, k] = pd.DataFrame(df_new[:, i, k]).interpolate().values.reshape(len(date_range))
                print("interpolate done")
                return df_new
            else:
                return df_new