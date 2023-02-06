import pytest
import pandas as pd


@pytest.fixture(scope='module')
def data_train():
    """Get customers processed train data to feed into the tests"""
    return pd.read_csv("./data/processed/train_feature_engineering.csv", index_col=[0])


@pytest.fixture(scope='module')
def data_test():
    """Get customers processed test data to feed into the tests"""
    return pd.read_csv("./data/processed/test_feature_engineering.csv", index_col=[0])


def test_train_duplicates(data_train):
    """Test if the train duplicated dataframe is empty --> no duplicates"""
    duplicates = data_train[data_train.duplicated()]
    assert duplicates.empty


def test_test_duplicates(data_test):
    """Test if the test duplicated dataframe is empty --> no duplicates"""
    duplicates = data_test[data_test.duplicated()]
    assert duplicates.empty


def test_train_target_col(data_train):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data_train.columns


def test_train_test_sizes(data_train, data_test):
    """Check that train and test dataframe have the same columns (but target)"""
    train_size = data_train.drop(columns='TARGET').shape[1]
    test_size = data_test.shape[1]
    assert train_size == test_size

