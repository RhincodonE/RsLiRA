import os
import glob
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
from tqdm import tqdm
from utils import calculate_log_odds, fit_normal_distribution, load_combined_log_odds_user_level, load_combined_log_odds_item_level, save_distribution_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon-DigitalMusic')
    parser.add_argument('--distribution_path', type=str, default='./distribution')
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.distribution_path):
        os.makedirs(args.distribution_path)

    # 1. Compute user-level distribution from user-data/shadow
    user_in_log_odds = load_combined_log_odds_user_level('./user-data/shadow', 'in')
    user_out_log_odds = load_combined_log_odds_user_level('./user-data/shadow', 'out')

    in_mean_user, in_variance_user = fit_normal_distribution(user_in_log_odds)
    out_mean_user, out_variance_user = fit_normal_distribution(user_out_log_odds)

    save_distribution_results(
        os.path.join(args.distribution_path, 'distribution_user_shadow.csv'),
        in_mean_user, in_variance_user, out_mean_user, out_variance_user
    )

    # 2. Compute user-item-level distribution from item-data/shadow
    item_in_log_odds = load_combined_log_odds_item_level('./item-data/shadow', 'in')
    item_out_log_odds = load_combined_log_odds_item_level('./item-data/shadow', 'out')

    in_mean_item, in_variance_item = fit_normal_distribution(item_in_log_odds)
    out_mean_item, out_variance_item = fit_normal_distribution(item_out_log_odds)

    save_distribution_results(
        os.path.join(args.distribution_path, 'distribution_item_shadow.csv'),
        in_mean_item, in_variance_item, out_mean_item, out_variance_item
    )
    user_in_log_odds = load_combined_log_odds_user_level('./user-data/target', 'in')
    user_out_log_odds = load_combined_log_odds_user_level('./user-data/target', 'out')

    in_mean_user, in_variance_user = fit_normal_distribution(user_in_log_odds)
    out_mean_user, out_variance_user = fit_normal_distribution(user_out_log_odds)

    save_distribution_results(
        os.path.join(args.distribution_path, 'distribution_user_target.csv'),
        in_mean_user, in_variance_user, out_mean_user, out_variance_user
    )

    # 2. Compute user-item-level distribution from item-data/shadow
    item_in_log_odds = load_combined_log_odds_item_level('./item-data/target', 'in')
    item_out_log_odds = load_combined_log_odds_item_level('./item-data/shadow', 'out')

    in_mean_item, in_variance_item = fit_normal_distribution(item_in_log_odds)
    out_mean_item, out_variance_item = fit_normal_distribution(item_out_log_odds)

    save_distribution_results(
        os.path.join(args.distribution_path, 'distribution_item_target.csv'),
        in_mean_item, in_variance_item, out_mean_item, out_variance_item
    )
if __name__ == "__main__":
    main()
