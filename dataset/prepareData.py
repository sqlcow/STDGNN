import argparse
import configparser
from dataset.prepareData2D import prepareData2D
import numpy as np




# read the parameters from the configuration file for 2D data
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='../configurations/BOSTON_multigcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
print(f'Reading configuration file {args.config} was successful!')
graph_signal_matrix_filename_2D=config.get('2D_data','graph_signal_matrix_filename_2D')
num_of_hours=config.get('2D_data','num_of_hours')
num_of_days=config.get('2D_data','num_of_days')
num_of_weeks=config.get('2D_data','num_of_weeks')
num_of_months=config.get('2D_data','num_of_months')
num_of_years=config.get('2D_data','num_of_years')
end_of_months=config.get('2D_data','end_of_months')
start_date=config.get('2D_data','start_date')
end_date=config.get('2D_data','end_date')
linear_interpolation=config.get('2D_data','linear_interpolation')

# Whether prepareData2D is selected, if not, the user needs to provide a preprocessed data set in PEMS format
if_2D=config.get('2D_data','if_2D')
if if_2D=='True':
    print("You choose the prepareData2D")
    prepareData2D=prepareData2D(graph_signal_matrix_filename_2D,num_of_hours,num_of_days,num_of_weeks,num_of_months,num_of_years,end_of_months,start_date,end_date,linear_interpolation)
    data=prepareData2D.prepareData2D
    #Save processed data
    #np.savez_compressed('../data/BOSTON/BOSTON1.npz', data=data)
else:
    print("YOU choose the prepareData like PEMS")

# read the parameters from the configuration file for generate the training data
adj_filename=config.get('Data','adj_filename')
graph_signal_matrix_filename=config.get('Data','graph_signal_matrix_filename')
variables_of_predicted_value=config.get('Modify','variables_of_predicted')
variables_of_training=config.get('Modify','variables_of_train')

# generate training data
if if_2D=='True':
    data=data
else:
    data=np.load(graph_signal_matrix_filename)['data']
    print(f'data shape ={data.shape}')