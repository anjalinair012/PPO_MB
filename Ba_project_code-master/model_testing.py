import pandas as pd

def get_variance(df):
    variance = df.var()
    std_dev = df.std()
    print(variance)


im_file = pd.read_csv("../EDITED-MATERIAL/175-FIX.csv",
                           index_col=False)  # normal_walking_data_edit2#ccw_data_walk_edit         175-FIX
get_variance(im_file)

