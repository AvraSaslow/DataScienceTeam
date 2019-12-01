import pandas as pd


pd.options.display.expand_frame_repr = False

epl = pd.read_csv('Data/EPL.csv')
print(epl.head())