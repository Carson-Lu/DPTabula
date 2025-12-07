from argparse import ArgumentParser
from data_split import data_split

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_all", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    data_split(args.data_all, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
