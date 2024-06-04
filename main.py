#!/usr/bin/env python3


from sdlabs.utility import *
from sdlabs.campaign import *
import sys, argparse


def main(argv):

    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_file', type=str)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
    parser.add_argument('--MAE', dest='MAE', action='store_true', default=True)
    parser.add_argument('--noMAE', dest='noMAE', action='store_true', default=False)
    args = parser.parse_args()
    if args.noMAE:
        args.MAE = False

    # Read the input file
    campaign_list = read_campaign_list(args.input_file)

    # Run each requested campaign
    # If MAE is requested, read the data back in from the output file and calculate the MAE
    for campaign in campaign_list:
        campaign.run(verbose=args.verbose)
        if args.MAE:
            campaign.calculate_MAE()

    return


if __name__ == '__main__':
    main(sys.argv[1:])