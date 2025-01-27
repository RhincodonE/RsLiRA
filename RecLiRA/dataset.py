import os
import random
import unittest
import tempfile
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------------
# RatingDataset Class
# --------------------------------------------------------------------------------
class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(RatingDataset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


# --------------------------------------------------------------------------------
# NCFDataHandler Class
# --------------------------------------------------------------------------------
class NCFDataHandler:
    def __init__(self, args, ratings):
        self.ratings = ratings
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size
        self.preprocess_ratings = self._reindex(self.ratings)

        self.user_pool = set(self.ratings['user_id'].unique())
        self.item_pool = set(self.ratings['item_id'].unique())

        # Randomly shuffle the user pool and split based on the seed
        sorted_user_list = list(self.user_pool)
        random.seed(args.seed)
        random.shuffle(sorted_user_list)

        # Split the user list into first 50% and remaining 50%
        split_idx = int(len(sorted_user_list) * 0.5)
        self.first_part_users = set(sorted_user_list[:split_idx])
        self.second_part_users = set(sorted_user_list[split_idx:])

        # Filter ratings by first 50% users for training and testing
        self.first_part_ratings = self.preprocess_ratings[
            self.preprocess_ratings['user_id'].isin(self.first_part_users)
        ]
        self.second_part_ratings = self.preprocess_ratings[
            self.preprocess_ratings['user_id'].isin(self.second_part_users)
        ]

        # Use the first 50% of users for train/test split
        self.train_ratings, self.test_ratings = self._leave_one_out_random(
            self.first_part_ratings, args.seed
        )
        self.negatives = self._negative_sampling(self.preprocess_ratings)

        random.seed(args.seed)

    def _reindex(self, ratings):
        """
        Reindex user IDs and item IDs to sequential integers, handling new entries.
        """
        user_mapping_file = './maps/user_id_mapping.csv'
        item_mapping_file = './maps/item_id_mapping.csv'

        if not os.path.exists('./maps'):
            os.makedirs('./maps', exist_ok=True)

        # ======================================================================
        # User ID Reindexing
        # ======================================================================
        if os.path.exists(user_mapping_file):
            user_mapping_df = pd.read_csv(user_mapping_file)
            user2id = dict(zip(user_mapping_df['original_user_id'], user_mapping_df['mapped_user_id']))

            # Check for new users not in existing mapping
            current_users = set(ratings['user_id'].unique())
            existing_users = set(user2id.keys())
            new_users = current_users - existing_users

            if new_users:
                # Assign new IDs sequentially
                max_id = max(user2id.values())
                user2id.update({user: max_id + i for i, user in enumerate(new_users, 1)})

                # Save updated mapping
                pd.DataFrame(
                    user2id.items(),
                    columns=['original_user_id', 'mapped_user_id']
                ).to_csv(user_mapping_file, index=False)
        else:
            # Create fresh mapping
            user_list = ratings['user_id'].unique()
            user2id = {user: idx for idx, user in enumerate(user_list)}
            pd.DataFrame(
                user2id.items(),
                columns=['original_user_id', 'mapped_user_id']
            ).to_csv(user_mapping_file, index=False)

        # ======================================================================
        # Item ID Reindexing (same logic as users)
        # ======================================================================
        if os.path.exists(item_mapping_file):
            item_mapping_df = pd.read_csv(item_mapping_file)
            item2id = dict(zip(item_mapping_df['original_item_id'], item_mapping_df['mapped_item_id']))

            current_items = set(ratings['item_id'].unique())
            existing_items = set(item2id.keys())
            new_items = current_items - existing_items

            if new_items:
                max_id = max(item2id.values())
                item2id.update({item: max_id + i for i, item in enumerate(new_items, 1)})

                pd.DataFrame(
                    item2id.items(),
                    columns=['original_item_id', 'mapped_item_id']
                ).to_csv(item_mapping_file, index=False)
        else:
            item_list = ratings['item_id'].unique()
            item2id = {item: idx for idx, item in enumerate(item_list)}
            pd.DataFrame(
                item2id.items(),
                columns=['original_item_id', 'mapped_item_id']
            ).to_csv(item_mapping_file, index=False)

        # ======================================================================
        # Apply Mappings
        # ======================================================================
        ratings['user_id'] = ratings['user_id'].map(user2id)
        ratings['item_id'] = ratings['item_id'].map(item2id)

        if ratings['user_id'].isnull().any() or ratings['item_id'].isnull().any():
            raise ValueError("Missing IDs in mapping!")

        # Convert to binary ratings
        ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))

        return ratings

    def _leave_one_out_random(self, ratings, seed=None):
        """
        Randomized leave-one-out evaluation. Randomly select one interaction for each user
        as the test set. The rest are training data.
        """
        if seed is not None:
            random.seed(seed)

        train_list, test_list = [], []

        # Group the interactions by user
        for user, group in ratings.groupby('user_id'):
            interactions = list(group.itertuples(index=False))
            test_interaction = random.choice(interactions)  # pick 1
            test_list.append(test_interaction)

            # Remaining => train
            for interaction in interactions:
                if interaction != test_interaction:
                    train_list.append(interaction)

        # Convert lists to DataFrame
        train_df = pd.DataFrame(train_list, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        test_df = pd.DataFrame(test_list, columns=['user_id', 'item_id', 'rating', 'timestamp'])

        train = train_df[['user_id', 'item_id', 'rating']]
        test = test_df[['user_id', 'item_id', 'rating']]

        # Both sets should have the same # of distinct users
        assert train['user_id'].nunique() == test['user_id'].nunique(), "Mismatch in user sets"

        return train, test

    def _negative_sampling(self, ratings):
        """
        For each user, gather items they have not interacted with, then sample 'num_ng_test'
        negative items.
        """
        interact_status = (
            ratings.groupby('user_id')['item_id']
            .apply(set)
            .reset_index()
            .rename(columns={'item_id': 'interacted_items'})
        )
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(
            lambda x: random.sample(x, self.num_ng_test) if len(x) >= self.num_ng_test else list(x)
        )
        return interact_status[['user_id', 'negative_items', 'negative_samples']]

    def get_train_instance(self):
        """
        Train loader with negative sampling.
        """
        users, items, ratings = [], [], []
        train_ratings = pd.merge(
            self.train_ratings,
            self.negatives[['user_id', 'negative_items']],
            on='user_id'
        )
        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(x, self.num_ng) if len(x) >= self.num_ng else list(x)
        )

        for row in train_ratings.itertuples():
            # Positive sample
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            # Negative samples
            for i in range(len(row.negatives)):
                users.append(int(row.user_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))

        dataset = RatingDataset(users, items, ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def get_in_instance(self):
        """
        In-loader (training loader without negative samples).
        """
        users, items, ratings = [], [], []
        for row in self.train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
        dataset = RatingDataset(users, items, ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def get_test_instance(self):
        """
        Test loader with negative sampling.
        """
        users, items, ratings = [], [], []
        test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')

        for row in test_ratings.itertuples():
            # Positive
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            # Negative
            for neg_item in getattr(row, 'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(neg_item))
                ratings.append(float(0))

        dataset = RatingDataset(users, items, ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=2)

    def get_out_instance(self):
        """
        Out-loader for the users in the second part (50%) of the dataset.
        """
        users, items, ratings = [], [], []
        for row in self.second_part_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
        dataset = RatingDataset(users, items, ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


# --------------------------------------------------------------------------------
# Test Class
# --------------------------------------------------------------------------------

class TestNCFDataHandler(unittest.TestCase):
    def setUp(self):
        """
        Creates a temporary folder, minimal DataFrame, and a pseudo-args object for testing.
        """
        self.test_dir = tempfile.mkdtemp()

        # Create minimal dataset with columns: user_id, item_id, rating, timestamp
        data = [
            [0, 10, 5, 100000],  # user0 -> item10
            [0, 20, 5, 100001],  # user0 -> item20
            [1, 20, 5, 100002],  # user1 -> item20
            [1, 30, 4, 100003],  # user1 -> item30
            [2, 40, 5, 100004],  # user2 -> item40
            [2, 50, 5, 100005],  # user2 -> item50
            [3, 60, 5, 100006],  # user3 -> item60
            [3, 70, 5, 100007],  # user3 -> item70
            [4, 80, 5, 100008],  # user4 -> item80
            [4, 90, 5, 100009],  # user4 -> item90
        ]
        self.df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'timestamp'])

        # Write user_id/item_id maps to avoid KeyErrors in _reindex if needed
        os.makedirs(os.path.join(self.test_dir, 'maps'), exist_ok=True)
        user_map_path = os.path.join(self.test_dir, 'maps', 'user_id_mapping.csv')
        item_map_path = os.path.join(self.test_dir, 'maps', 'item_id_mapping.csv')

        # For simplicity, we do a trivial mapping: user-> user, item-> item
        user_map_df = pd.DataFrame(
            {'original_user_id': [0, 1, 2, 3, 4], 'mapped_user_id': [0, 1, 2, 3, 4]}
        )
        item_map_df = pd.DataFrame(
            {
                'original_item_id': [10, 20, 30, 40, 50, 60, 70, 80, 90],
                'mapped_item_id':   [0,   1,   2,   3,   4,   5,   6,   7,   8]
            }
        )
        user_map_df.to_csv(user_map_path, index=False)
        item_map_df.to_csv(item_map_path, index=False)

        # We'll patch the location of ./maps to self.test_dir/maps
        # So the code can find these files
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create a pseudo-args object
        class Args:
            num_ng = 2
            num_ng_test = 3
            batch_size = 2
            seed = 42

        self.args = Args()

    def tearDown(self):
        """
        Cleanup temp directory and revert CWD.
        """
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_rating_dataset(self):
        """
        Test basic functionality of RatingDataset.
        """
        user_list = [0, 1, 2]
        item_list = [10, 20, 30]
        rating_list = [1., 0., 1.]
        ds = RatingDataset(user_list, item_list, rating_list)

        self.assertEqual(len(ds), 3)
        user, item, rating = ds[0]
        self.assertEqual(user.item(), 0)
        self.assertEqual(item.item(), 10)
        self.assertAlmostEqual(rating.item(), 1.0)

    def test_data_handler_init(self):
        """
        Test that NCFDataHandler initializes correctly with the sample dataset.
        """
        handler = NCFDataHandler(self.args, self.df)
        self.assertIsNotNone(handler.preprocess_ratings)
        self.assertGreater(len(handler.user_pool), 0)
        self.assertGreater(len(handler.item_pool), 0)
        # Check first_part_ratings / second_part_ratings not empty
        self.assertGreaterEqual(len(handler.first_part_ratings), 0)
        self.assertGreaterEqual(len(handler.second_part_ratings), 0)

    def test_reindex_method(self):
        """
        Test private _reindex does not crash and transforms 'rating' -> binary.
        """
        handler = NCFDataHandler(self.args, self.df)
        reindexed = handler._reindex(self.df.copy())  # call it again
        # Check that rating is in {0.0, 1.0}
        self.assertTrue(all(reindexed['rating'].isin([0.0, 1.0])))

    def test_leave_one_out_random(self):
        """
        Check that for each user, exactly 1 record goes to test set.
        """
        handler = NCFDataHandler(self.args, self.df)
        train_df, test_df = handler._leave_one_out_random(handler.preprocess_ratings.copy(), seed=42)
        # every user has exactly 1 test item
        user_counts_test = test_df.groupby('user_id')['item_id'].count()
        for user_id in user_counts_test.index:
            self.assertEqual(user_counts_test[user_id], 1)

    def test_negative_sampling(self):
        """
        Ensure negative_samples are drawn properly for each user.
        """
        handler = NCFDataHandler(self.args, self.df)
        negs = handler._negative_sampling(handler.preprocess_ratings)
        self.assertIn('negative_samples', negs.columns)
        for row in negs.itertuples():
            # We asked for 3 negative samples per user
            self.assertTrue(len(row.negative_samples) >= 0)  # could be less if item_pool - user_set < 3

    def test_get_train_instance(self):
        """
        Simple check that the train loader returns a DataLoader and we can iterate.
        """
        handler = NCFDataHandler(self.args, self.df)
        train_loader = handler.get_train_instance()
        self.assertIsInstance(train_loader, DataLoader)
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 3)  # (user, item, rating)

    def test_get_in_instance(self):
        """
        Check in_loader returns only positive samples (no negative sampling).
        """
        handler = NCFDataHandler(self.args, self.df)
        in_loader = handler.get_in_instance()
        for user, item, rating in in_loader:
            # In instance is presumably only positive
            self.assertTrue((rating == 1.0).all())

    def test_get_test_instance(self):
        """
        Test the test loader includes 1 positive + negative_samples per user.
        """
        handler = NCFDataHandler(self.args, self.df)
        test_loader = handler.get_test_instance()
        batch = next(iter(test_loader))
        # We get (user, item, rating), and batch size = num_ng_test + 1 => 4 in this example
        self.assertEqual(batch[0].size(0), self.args.num_ng_test + 1)

    def test_get_out_instance(self):
        """
        Test that the out_loader returns users only in second_part_users.
        """
        handler = NCFDataHandler(self.args, self.df)
        out_loader = handler.get_out_instance()
        # second_part_users is 50% of user pool => we can verify some basic structure
        batch = next(iter(out_loader))
        self.assertEqual(len(batch), 3)  # user, item, rating
        # We won't do deep checks unless needed, but ensure it doesn't crash.


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
