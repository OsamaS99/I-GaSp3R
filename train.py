from ingas.training import get_args_parser, main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InGaS Training Script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
