import sys
sys.path.append('/home/tehreem/nn-dataset')
import ab.nn.api as nn_dataset

try:
    print("Querying ResNet...")
    df = nn_dataset.data(nn='ResNet')
    if df.empty:
        print("DataFrame is empty!")
    else:
        print(f"Found {len(df)} rows.")
        code = df.iloc[0]['nn_code']
        print(f"Code length: {len(code)}")
        print("First 50 chars:", code[:50])
except Exception as e:
    print(f"API Error: {e}")
