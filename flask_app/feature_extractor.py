import re
import pandas as pd
from urllib.parse import urlparse

# EXACT feature list from dataset 3 (ORDER MATTERS)
FEATURE_COLUMNS = [
    'url_length', 'n_dots', 'n_hyphens', 'n_underline', 'n_slash',
    'n_questionmark', 'n_equal', 'n_at', 'n_and', 'n_exclamation',
    'n_space', 'n_tilde', 'n_comma', 'n_plus', 'n_asterisk',
    'n_hashtag', 'n_dollar', 'n_percent', 'n_redirection'
]

def extract_features(url: str) -> pd.DataFrame:
    features = {}

    features['url_length'] = len(url)
    features['n_dots'] = url.count('.')
    features['n_hyphens'] = url.count('-')
    features['n_underline'] = url.count('_')
    features['n_slash'] = url.count('/')
    features['n_questionmark'] = url.count('?')
    features['n_equal'] = url.count('=')
    features['n_at'] = url.count('@')
    features['n_and'] = url.count('&')
    features['n_exclamation'] = url.count('!')
    features['n_space'] = url.count(' ')
    features['n_tilde'] = url.count('~')
    features['n_comma'] = url.count(',')
    features['n_plus'] = url.count('+')
    features['n_asterisk'] = url.count('*')
    features['n_hashtag'] = url.count('#')
    features['n_dollar'] = url.count('$')
    features['n_percent'] = url.count('%')
    features['n_redirection'] = url.count('//') - 1

    # Create DataFrame with EXACT columns
    df = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]],
                      columns=FEATURE_COLUMNS)

    return df
