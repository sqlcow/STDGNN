import pandas as pd
import numpy as np
class prepareData2D:
    def __init__(self,graph_signal_matrix_filename,num_of_hours,num_of_days,num_of_weeks,num_of_months,num_of_years,end_of_months):
        self.graph_signal_matrix_filename=graph_signal_matrix_filename
        self.num_of_hours = int(num_of_hours)
        self.num_of_days = int(num_of_days)
        self.num_of_weeks = int(num_of_weeks)
        self.num_of_months = int(num_of_months)
        self.num_of_years = int(num_of_years)
        self.end_of_months = end_of_months
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
            raise Exception('The column name is wrong. The names of the first two columns of the 2D object need to be id and date_time.')

    def prepareData2D(self):
        df = pd.read_csv(self.graph_signal_matrix_filename)
        # Check whether the df object is legal
        self.check_column_names(df)
        if(self.end_of_months=='True'):
            pass
        else:
            earliest_date = df['date_time'].min()
            latest_date = df['date_time'].max()
            print(f"Automatically identify the start and end date pattern: earliest date is {earliest_date} latest date is {latest_date}")
            if earliest_date == latest_date:
                raise Exception("The dataset contains only one time step. Please check the dataset.")
            if (self.num_of_hours + self.num_of_days + self.num_of_weeks + self.num_of_months + self.num_of_years) == 0:
                raise Exception("All the time steps are 0. Please check the dataset.")
            delta = pd.DateOffset(hours=self.num_of_hours,days=self.num_of_days,weeks=self.num_of_weeks,  months=self.num_of_months, years=self.num_of_years)
            date_range = pd.date_range(earliest_date, latest_date, freq=delta)
            df['Recode'] = df.groupby('id').ngroup()
            date_time_list = date_range.tolist() * df['Recode'].max()
            df_new = pd.DataFrame({'date_time': date_time_list})
            df_new['Recode'] = np.repeat(np.arange(0, df['Recode'].max()), len(date_range))
            df['date_time'] = pd.to_datetime(df['date_time'])
            df_new = pd.merge(df_new, df, on=['Recode', 'date_time'], how='left')
            df_new = df_new.drop(['Recode','date_time','id'], axis=1)