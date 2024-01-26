import argparse
import configparser
from dataset.prepareData2D import prepareData2D
parser = argparse.ArgumentParser()

parser.add_argument("--config", default='../configurations/BOSTON_multigcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
print(f'Reading configuration file {args.config} was successful!')


# Whether prepareData2D is selected, if not, the user needs to provide a preprocessed data set in PEMS format
if_2D=config.get('2D_data','if_2D')
if if_2D=='True':
    print("You choose the prepareData2D")
    prepareData2D=prepareData2D(config.get('2D_data','2Dgraph_signal_matrix_filename'),
    config.get('2D_data', 'num_of_hours'),
    config.get('2D_data','num_of_days'),
    config.get('2D_data','num_of_weeks'),
    config.get('2D_data', 'num_of_months'),
    config.get('2D_data', 'num_of_years'),config.get('2D_data', 'end_of_months'))
    fuck_file_name=prepareData2D.prepareData2D()

else:
    print("YOU choose the prepareData like PEMS")