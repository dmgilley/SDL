#!/usr/bin/env python3


import sys, inspect, datetime, json, pickle, argparse
from SDL import material, strategy, utility, world


def main(argv):

    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_file', type=str)
    args = parser.parse_args()

    # Read the input file
    campaign_list = utility.read_campaign_list(args.input_file)

    # Run each requested campaign
    for campaign in campaign_list:
        campaign.run()

        # If MAE is requested, read the data back in from the output file and calculate the MAE
        if args.MAE:
            data = utility.read_output(
                campaign.filename,
                data=utility.SDLOutputData(
                    campaign.name, 
                    calc_stability=campaign.environment.experiments['Stability'].calc_stability
                    )
                )
            campaign.dump_to_MAE()
            for experiment_name in [k for k,v in campaign.environment.experiments.items() if v.category == 'processing']:
                reference_inputs = campaign.environment.experiments[experiment_name].get_input_space()[1]
                reference_targets = data.calc_stability(reference_inputs)
                MAE = {run: utility.get_cumulative_MAE(data,run,reference_inputs,reference_targets) for run in range(1,data.runs+1)}
                campaign.dump_to_MAE(MAE)

    return

if __name__ == '__main__':
    main(sys.argv[1:])