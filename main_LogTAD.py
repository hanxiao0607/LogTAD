import os
from utils import preprocessing, SlidingWindow
from utils.utils import set_seed, get_train_eval_iter
from argparse import ArgumentParser
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from model.LogTAD import LogTAD

def arg_parser():
    """
    Add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default='Thunderbird')
    parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default='BGL')
    parser.add_argument("--device", help="hardware device", default="cuda")
    parser.add_argument("--output_dir", metavar="DIR", help="output directory", default="/Dataset")
    parser.add_argument("--model_dir", metavar="DIR", help="model directory", default="/Dataset")
    parser.add_argument("--random_seed", help="random seed", default=42)
    parser.add_argument("--download_datasets", help="download datasets or not", default=0)

    # training parameters
    parser.add_argument("--max_epoch", help="epochs", default=100)
    parser.add_argument("--batch_size", help="batch size", default=1024)
    parser.add_argument("--lr", help="learning rate", default=0.001)
    parser.add_argument("--weight_decay", help="weight decay", default=1e-6)
    parser.add_argument("--eps", help="minimum center value", default=0.1)

    # word2vec parameters
    parser.add_argument("--emb_dim", help="word2vec vector size", default=300)

    # data preprocessing parameters
    parser.add_argument("--window_size", help="size of sliding window", default=20)
    parser.add_argument("--step_size", help="step size of sliding window", default=4)
    parser.add_argument("--train_size_s", help="source training size", default=100000)
    parser.add_argument("--train_size_t", help="target training size", default=1000)

    # LSTM parameters
    parser.add_argument("--hid_dim", help="hidden dimensions", default=128)
    parser.add_argument("--out_dim", help="output dimensions", default=2)
    parser.add_argument("--n_layers", help="layers of LSTM", default=2)
    parser.add_argument("--dropout", help="dropout", default=0.3)
    parser.add_argument("--bias", help="bias for LSTM", default=True)

    # gradient reversal parameters
    parser.add_argument("--alpha", help="alpha value for the gradient reversal layer", default=0.1)

    #test parameters
    parser.add_argument("--test_ratio", help="testing ratio", default=0.1)

    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    options = vars(args)

    set_seed(options["random_seed"])
    print(f"Set seed: {options['random_seed']}")
    if options["download_datasets"] == 1:
        preprocessing.parsing(options["source_dataset_name"], options["output_dir"])
        preprocessing.parsing(options["target_dataset_name"], options["output_dir"])

    path = "./Dataset"
    if len(os.listdir(path)) == 0:
        print("Please download the dataset first")
        return 1

    df_source = pd.read_csv(f'./Dataset/{options["source_dataset_name"]}.log_structured.csv')
    print(f'Reading source dataset: {options["source_dataset_name"]} dataset')
    df_target = pd.read_csv(f'./Dataset/{options["target_dataset_name"]}.log_structured.csv')
    print(f'Reading target dataset: {options["target_dataset_name"]} dataset')
    train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v = SlidingWindow.get_datasets(df_source, df_target, options)
    train_iter, test_iter = get_train_eval_iter(train_normal_s, train_normal_t)
    demo_logtad = LogTAD(options)
    print('here')
    demo_logtad.train_LogTAD(train_iter, test_iter, w2v)
    print('done')



if __name__ == "__main__":
    main()
