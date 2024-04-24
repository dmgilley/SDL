#!/usr/bin/env python3


import sys, inspect, datetime, json, pickle, argparse
from sdlabs.utility import *


def main(argv):

    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_file', type=str)
    parser.add_argument('--MAE', dest='MAE', action='store_true', default=True)
    parser.add_argument('--noMAE', dest='noMAE', action='store_true', default=False)
    args = parser.parse_args()
    if args.noMAE:
        args.MAE = False

    # Read the input file
    campaign_list = read_campaign_list(args.input_file)

    # Run each requested campaign
    for campaign in campaign_list:
        campaign.run()

        # If MAE is requested, read the data back in from the output file and calculate the MAE
        if args.MAE:
            campaign.run_and_dump_MAE()

    return

if __name__ == '__main__':
    main(sys.argv[1:])