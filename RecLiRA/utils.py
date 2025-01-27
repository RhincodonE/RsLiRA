import os
import unittest
import tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import glob
from collections import defaultdict
from scipy.stats import norm
# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------

def download_data(data_url, data_path):
    """
    Downloads data from the specified URL to data_path if it doesn't exist.
    """
    if not os.path.exists(data_path):
        os.system(f'wget -nc {data_url} -O {data_path}')
    else:
        print(f"Data already exists at {data_path}")

def load_data(file_path):
    """
    Loads the ML-1M dataset and filters out users with < 20 interactions.
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep="::", engine='python', names=column_names)

    user_interaction_counts = df['user_id'].value_counts()
    eligible_users = user_interaction_counts[user_interaction_counts >= 20].index
    filtered_df = df[df['user_id'].isin(eligible_users)]
    return filtered_df

def amazon_load(file_path, min_interactions=20):
    """
    Loads and processes the Amazon dataset, excluding users with fewer than `min_interactions`.
    """
    df = pd.read_csv(file_path, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_interaction_counts = df['user_id'].value_counts()
    eligible_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
    filtered_df = df[df['user_id'].isin(eligible_users)]
    return filtered_df

def collect_probability_differences(model, data_loader, device):
    """
    Collects absolute probability differences: abs(2P - 1) for each sample in data_loader.
    """
    probability_differences = []
    model.eval()

    with torch.no_grad():
        for user, item, label in data_loader:
            user = user.to(device)
            item = item.to(device)

            predictions = model(user, item)
            prob_diff = torch.abs(2 * predictions - 1)
            probability_differences.extend(prob_diff.cpu().numpy())

    return np.array(probability_differences)

def hit(ng_item, pred_items):
    """
    Checks if the ground-truth item is in the top-k predictions.
    """
    return 1 if ng_item in pred_items else 0

def ndcg(ng_item, pred_items):
    """
    Computes normalized discounted cumulative gain for a single user.
    """
    if ng_item in pred_items:
        index = pred_items.index(ng_item)
        return 1 / np.log2(index + 2)
    return 0

def metrics(model, test_loader, top_k, device):
    """
    Computes average HR and NDCG over the test_loader.
    """
    HR, NDCG = [], []
    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        ng_item = item[0].item()  # leave-one-out evaluation
        HR.append(hit(ng_item, recommends))
        NDCG.append(ndcg(ng_item, recommends))

    return np.mean(HR), np.mean(NDCG)

def collect_probability_differences_by_user(model, data_loader, device, flag="sum"):
    """
    Collect and aggregate probability differences by user ID, also store user-item probabilities.
    """
    user_prob_diffs = {}
    user_counts = {}
    user_item_diff = {}

    model.eval()
    progress_bar = tqdm(data_loader, desc="Collecting probability differences", leave=False)

    with torch.no_grad():
        for user, item, _ in progress_bar:
            user = user.to(device)
            item = item.to(device)

            predictions = model(user, item)
            prob_diff = torch.abs(2 * predictions - 1)

            for i, u in enumerate(user.cpu().numpy()):
                it = item[i].cpu().numpy().item()

                if u not in user_prob_diffs:
                    user_prob_diffs[u] = 0.0
                    user_counts[u] = 0
                    user_item_diff[u] = []

                user_prob_diffs[u] += prob_diff[i].item()
                user_counts[u] += 1
                user_item_diff[u].append((it, predictions[i].item()))

    if flag == "average":
        for uid in user_prob_diffs:
            user_prob_diffs[uid] /= user_counts[uid]

    return user_prob_diffs, user_item_diff

def calculate_log_odds(prob_array):
    """
    Calculates log-odds for a given array of probabilities.
    """
    epsilon = 1e-5
    prob_array = np.clip(prob_array, epsilon, 1 - epsilon)
    return np.log(prob_array / (1 - prob_array))

def calculate_log_odds_sum(input_array):
    """
    Transforms input from (0, +inf) to (-inf, +inf) using the natural logarithm.
    """
    epsilon = 1e-5  # Small constant to avoid issues with very small values
    input_array = np.clip(input_array, epsilon, None)  # Clip to avoid zero or negative inputs
    return np.log(input_array)

def fit_normal_distribution(data_array):
    """
    Returns the mean and variance of the data_array.
    """
    mean = np.mean(data_array)
    variance = np.var(data_array)
    return mean, variance

def mkdir(directory):
    """
    Creates a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def build_adj_matrix(num_users, num_items, interactions):
    # Separate user and item indices
    user_indices = interactions['user_id'].values
    item_indices = interactions['item_id'].values + num_users  # Offset item indices

    # Define values array with ones
    values = np.ones(len(user_indices))

    # Create rows and columns for adjacency matrix
    rows = np.concatenate([user_indices, item_indices])
    cols = np.concatenate([item_indices, user_indices])
    data = np.concatenate([values, values])

    # Build the adjacency matrix
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))

    # Normalize the adjacency matrix
    rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj_matrix = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)
    norm_adj_matrix = norm_adj_matrix.tocoo()

    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Number of non-zero entries: {adj_matrix.nnz}")

    return norm_adj_matrix


def load_combined_log_odds_user_level(directory, prefix):
    combined_log_odds = []
    file_pattern = f'{directory}/{prefix}_*.csv'
    file_paths = glob.glob(file_pattern)

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # Expecting a column named 'prob_log_odds'
        if 'prob_log_odds' in df.columns:
            values = df['prob_log_odds'].values
        else:
            values = np.loadtxt(file_path, delimiter=',')
        combined_log_odds.extend(values)
        print(f'Loaded {prefix} log odds from {file_path}')

    return np.array(combined_log_odds)

def load_combined_log_odds_item_level(directory, prefix):
    combined_log_odds = []
    file_pattern = f'{directory}/{prefix}_*.csv'
    file_paths = glob.glob(file_pattern)

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if 'prob_log_odds' not in df.columns:
            raise ValueError(f"File {file_path} does not have 'prob_log_odds' column.")
        values = df['prob_log_odds'].values
        combined_log_odds.extend(values)
        print(f'Loaded {prefix} item-level log odds from {file_path}')

    return np.array(combined_log_odds)

def save_distribution_results(filename, in_mean, in_variance, out_mean, out_variance):
    df = pd.DataFrame({
        'in_mean': [in_mean],
        'in_variance': [in_variance],
        'out_mean': [out_mean],
        'out_variance': [out_variance]
    })
    df.to_csv(filename, index=False)
    print(f"Saved distribution data to {filename}")

def compute_user_specific_thresholds(out_mean_loaded, out_variance_loaded, user_data):
    """
    Compute user-level thresholds and metrics (TPR, FPR) for each user.
    """
    out_std = np.sqrt(out_variance_loaded)
    user_metrics = {}

    for uid, vals in user_data.items():
        in_obs = np.array(vals["in"])
        out_obs = np.array(vals["out"])

        # Compute CDF values for this user
        in_cdf = norm.cdf(in_obs, loc=out_mean_loaded, scale=out_std) if len(in_obs) > 0 else np.array([])
        out_cdf = norm.cdf(out_obs, loc=out_mean_loaded, scale=out_std) if len(out_obs) > 0 else np.array([])

        # If the user has no out observations or no in observations, set NaN
        if len(out_cdf) == 0 or len(in_cdf) == 0:
            user_metrics[uid] = (np.nan, np.nan, np.nan)
            continue

        # threshold at 99th percentile of out_cdf to ensure FPR ≤ 0.01
        threshold = np.percentile(out_cdf, 99)

        TPR_user = np.sum(in_cdf > threshold) / len(in_cdf) if len(in_cdf) > 0 else np.nan
        FPR_user = np.sum(out_cdf > threshold) / len(out_cdf) if len(out_cdf) > 0 else np.nan

        user_metrics[uid] = (threshold, TPR_user, FPR_user)

    return user_metrics

def compute_user_item_specific_thresholds(out_mean_loaded, out_variance_loaded, user_item_data):
    """
    Compute item-level thresholds and metrics (TPR, FPR) for each user-item pair.
    """
    out_std = np.sqrt(out_variance_loaded)
    item_metrics = []

    # user_item_data[user_id][item_id] = {"in": [...], "out": [...]}
    for uid, item_dict in user_item_data.items():
        for it, vals in item_dict.items():
            in_obs = np.array(vals["in"])
            out_obs = np.array(vals["out"])

            in_cdf = norm.cdf(in_obs, loc=out_mean_loaded, scale=out_std) if len(in_obs) > 0 else np.array([])
            out_cdf = norm.cdf(out_obs, loc=out_mean_loaded, scale=out_std) if len(out_obs) > 0 else np.array([])

            # If no out or no in observations, set NaN
            if len(out_cdf) == 0 or len(in_cdf) == 0:
                # user_id, item_id, TPR, FPR as NaNs
                item_metrics.append((uid, it, np.nan, np.nan))
                continue

            # threshold at 99th percentile of out_cdf
            threshold = np.percentile(out_cdf, 99)

            TPR_user = np.sum(in_cdf > threshold) / len(in_cdf) if len(in_cdf) > 0 else np.nan
            FPR_user = np.sum(out_cdf > threshold) / len(out_cdf) if len(out_cdf) > 0 else np.nan

            item_metrics.append((uid, it, TPR_user, FPR_user))

    return item_metrics

def load_user_distribution(args):
    """
    Load user-level in_mean, in_variance, out_mean, and out_variance from distribution_user.csv
    """
    csv_filename = f"{args.distribution_path}/distribution_user_shadow.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        in_mean = df['in_mean'].values[0]
        in_variance = df['in_variance'].values[0]
        out_mean = df['out_mean'].values[0]
        out_variance = df['out_variance'].values[0]
        print(f"Loaded user-level distribution data from {csv_filename}")
        return in_mean, in_variance, out_mean, out_variance
    else:
        print(f"User-level distribution file {csv_filename} does not exist.")
        return None, None, None, None

def load_item_distribution(args):
    """
    Load item-level in_mean, in_variance, out_mean, and out_variance from distribution_item.csv
    """
    csv_filename = f"{args.distribution_path}/distribution_item_shadow.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        in_mean = df['in_mean'].values[0]
        in_variance = df['in_variance'].values[0]
        out_mean = df['out_mean'].values[0]
        out_variance = df['out_variance'].values[0]
        print(f"Loaded item-level distribution data from {csv_filename}")
        return in_mean, in_variance, out_mean, out_variance
    else:
        print(f"Item-level distribution file {csv_filename} does not exist.")
        return None, None, None, None

def load_all_pairs(directory):
    """
    Load user-level pairs from user-data/target/.
    user_data[user_id] = {"in": [...], "out": [...]}
    """
    in_files = glob.glob(os.path.join(directory, "in_*.csv"))
    user_data = defaultdict(lambda: {"in": [], "out": []})

    for in_file in in_files:
        out_file = in_file.replace("in_", "out_")
        if not os.path.exists(out_file):
            continue

        in_df = pd.read_csv(in_file)
        out_df = pd.read_csv(out_file)

        for uid, val in zip(in_df['user_id'], in_df['prob_log_odds']):
            user_data[uid]["in"].append(val)

        for uid, val in zip(out_df['user_id'], out_df['prob_log_odds']):
            user_data[uid]["out"].append(val)

    return user_data

def load_all_item_pairs(directory):
    """
    Load item-level pairs from item-data/target/.
    user_item_data[user_id][item_id] = {"in": [...], "out": [...]}
    """
    in_files = glob.glob(os.path.join(directory, "in_*.csv"))
    user_item_data = defaultdict(lambda: defaultdict(lambda: {"in": [], "out": []}))

    for in_file in in_files:
        out_file = in_file.replace("in_", "out_")
        if not os.path.exists(out_file):
            continue

        in_df = pd.read_csv(in_file)
        out_df = pd.read_csv(out_file)

        # For each row in in_df
        for uid, it, val in zip(in_df['user_id'], in_df['item_id'], in_df['prob_log_odds']):
            user_item_data[uid][it]["in"].append(val)

        # For each row in out_df
        for uid, it, val in zip(out_df['user_id'], out_df['item_id'], out_df['prob_log_odds']):
            user_item_data[uid][it]["out"].append(val)

    return user_item_data

# --------------------------------------------------------------------------------
# Minimal Model Stub for Testing
# --------------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """
    A minimal PyTorch model that returns a "probability-like" output.
    This allows testing of functions that require a model to generate predictions.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # just 2 -> 1

    def forward(self, user, item):
        # user and item are both single-column tensors
        # We can just stack them or do something trivial for demonstration
        x = torch.stack([user.float(), item.float()], dim=1)
        out = torch.sigmoid(self.linear(x)).squeeze()
        return out


# --------------------------------------------------------------------------------
# Test Class
# --------------------------------------------------------------------------------

class TestUtils(unittest.TestCase):

    def setUp(self):
        """
        Creates a temporary directory and test CSV data for load tests.
        Also sets up a model and a DataLoader for testing model-based functions.
        """
        self.test_dir = tempfile.mkdtemp()

        # Create sample data for load_data or amazon_load
        # We'll create a minimal dataset with user_id, item_id, rating, timestamp
        # Some users have <20 interactions to test the filtering.
        data = [
            [1, 10, 4, 1111111111],
            [1, 20, 5, 1111111112],
            [2, 20, 3, 1111111113],
            [2, 30, 4, 1111111114],
            # user_id=3 will have 21 interactions to pass the threshold
        ]
        # Add 21 lines for user_id=3
        for i in range(21):
            data.append([3, 100 + i, 5, 1111111115 + i])

        self.test_file_ml = os.path.join(self.test_dir, "ratings_ml.dat")
        # ML-1M style uses "::" as a separator
        with open(self.test_file_ml, "w") as f:
            for row in data:
                f.write("::".join(map(str, row)) + "\n")

        # Amazon style uses CSV with no header
        self.test_file_amz = os.path.join(self.test_dir, "ratings_amz.csv")
        # We'll re-use the same data (just in CSV format).
        df_amz = pd.DataFrame(data)
        df_amz.to_csv(self.test_file_amz, header=False, index=False)

        # Create a simple model and data for the PyTorch-based tests
        self.model = SimpleModel()

        # We'll create a small DataLoader that yields (user, item, label)
        # For demonstration: let's create a small artificial dataset
        # user, item, label can just be 1 or 0
        self.device = torch.device('cpu')
        user_tensor = torch.tensor([1, 2, 3, 2, 3])
        item_tensor = torch.tensor([10, 20, 30, 99, 100])
        label_tensor = torch.tensor([1., 0., 1., 0., 1.])

        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, label_tensor)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    def tearDown(self):
        """
        Remove the temporary directory after tests.
        """
        import shutil
        shutil.rmtree(self.test_dir)

    def test_download_data(self):
        """
        Just tests that the function runs without error if file already exists
        or attempts to wget if not.
        """
        # We won't actually test the network call here, just run the function
        download_data("http://example.com/fakefile.txt", os.path.join(self.test_dir, "testfile.txt"))
        self.assertTrue(True)  # If no error, we pass

    def test_load_data(self):
        """
        Should filter out users with fewer than 20 interactions.
        We have user_id=1,2 with <20; user_id=3 with 21 interactions => only user_id=3 remains.
        """
        df = load_data(self.test_file_ml)
        unique_users = df['user_id'].unique()
        self.assertEqual(len(unique_users), 1)
        self.assertEqual(unique_users[0], 3)

    def test_amazon_load(self):
        """
        Similar test for amazon_load but using the CSV file.
        """
        df = amazon_load(self.test_file_amz, min_interactions=20)
        unique_users = df['user_id'].unique()
        self.assertEqual(len(unique_users), 1)
        self.assertEqual(unique_users[0], 3)

    def test_collect_probability_differences(self):
        """
        Test that the function returns the correct shape and type of array.
        """
        probs = collect_probability_differences(self.model, self.data_loader, self.device)
        # Data loader has 5 items total, so we expect an array of length 5
        self.assertEqual(len(probs), 5)
        self.assertTrue(isinstance(probs, np.ndarray))

    def test_hit(self):
        """
        Test the hit function.
        """
        # ground truth is 10, predictions are [20,10] => index 1 => hit
        self.assertEqual(hit(10, [20, 10]), 1)
        # ground truth 99 not in [20,10] => no hit
        self.assertEqual(hit(99, [20, 10]), 0)

    def test_ndcg(self):
        """
        Test the ndcg function.
        """
        # ground truth is 10. If predictions is [20,10], index=1 => NDCG=1/log2(1+2)=1/log2(3)
        val = ndcg(10, [20, 10])
        self.assertAlmostEqual(val, 1/np.log2(3))

        # If item not found => 0
        val2 = ndcg(99, [20, 10])
        self.assertEqual(val2, 0)

    def test_metrics(self):
        """
        Test the metrics function (HR, NDCG).
        We just check that we get valid numeric results back.
        """
        hr, n = metrics(self.model, self.data_loader, top_k=2, device=self.device)
        self.assertTrue(0.0 <= hr <= 1.0)
        self.assertTrue(0.0 <= n <= 1.0)

    def test_collect_probability_differences_by_user(self):
        """
        Test the user-level probability difference aggregator.
        """
        user_prob, user_item_diff = collect_probability_differences_by_user(
            self.model,
            self.data_loader,
            self.device,
            flag="average"
        )
        # We have user IDs = [1,2,3], so we expect 3 keys
        self.assertEqual(len(user_prob), 3)
        self.assertEqual(len(user_item_diff), 3)
        # Each user should have at least 1 or more items
        for k, v in user_item_diff.items():
            self.assertTrue(len(v) > 0)

    def test_calculate_log_odds(self):
        """
        Test the log-odds calculation.
        """
        arr = np.array([0.01, 0.5, 0.99])
        logs = calculate_log_odds(arr)
        self.assertEqual(len(logs), 3)
        # Just check we get finite numbers
        self.assertTrue(np.isfinite(logs).all())

    def test_fit_normal_distribution(self):
        """
        Test the fit_normal_distribution function.
        """
        data = np.array([1, 2, 3, 4, 5])
        mean, var = fit_normal_distribution(data)
        self.assertEqual(mean, 3.0)
        self.assertEqual(var, 2.0)

    def test_mkdir(self):
        """
        Test the mkdir function to ensure it creates the directory.
        """
        test_subdir = os.path.join(self.test_dir, "subdir")
        mkdir(test_subdir)
        self.assertTrue(os.path.exists(test_subdir))


# --------------------------------------------------------------------------------
# Main (for running tests)
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
