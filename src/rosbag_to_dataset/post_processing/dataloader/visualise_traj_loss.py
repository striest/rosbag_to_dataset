import pandas as pd
import matplotlib.pyplot as plt
import ast
from os.path import join 
import argparse
from rosbag_to_dataset.util.os_util import maybe_mkdir

def plot_loss(df,index="Epoch", losses = [], title = "", results_fp = ""):
    cols = [index] + losses
    df[cols].set_index(index).plot(figsize=(20, 10))
    plt.grid(which='both')
    plt.title(title)
    plt.savefig(join(results_fp,title+".png"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_fp', type=str, required=True, help='The root folder for the log')
    parser.add_argument('--name', type=str, required=False,default='log.csv', help='The name of the log file. By default - log.csv')
    args = parser.parse_args()
    log_fp=args.root_fp
    results_fp = join(log_fp, "plots")
    maybe_mkdir(results_fp,force = False)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    df = pd.read_csv(join(log_fp,args.name))
    df = df.rename(columns={"/Epoch": "Epoch"})
    drop_keywords = ['Debug','Timing']
    rem_cols = [i for j in drop_keywords for i in df.columns if j in i ]
    df.pop('Unnamed: 0')
    df = df.drop(columns=rem_cols)
    final_rmse_list = [i for i in df.columns if 'RMSE' in i]
    plot_loss(df,losses=final_rmse_list,title="Train and Eval RMSE",results_fp=join(log_fp,"plots"))
    final_loss_list = [i for i in df.columns if 'Loss' in i]
    for i in final_loss_list:
        plot_loss(df,losses=[i],title=i.split('/')[-1],results_fp=join(log_fp,"plots"))