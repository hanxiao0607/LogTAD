from utils import preprocessing
from utils.utils import set_seed
from argparse import ArgumentParser

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
    print("Set seed")
    preprocessing.parsing(options["source_dataset_name"], options["output_dir"])
    preprocessing.parsing(options["target_dataset_name"], options["output_dir"])
    print('done')


if __name__ == "__main__":
    main()