def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from pathlib import Path

    sns.set()

    DATA_DIR = Path('/content/drive/MyDrive/Work/Delivery/Current/Earthquake_damage/data')
    SUBMISSIONS_DIR = Path('drive/MyDrive/Work/Delivery/Current/Earthquake_damage/submissions')

    from google.colab import drive
    drive.mount('/content/drive')

    train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')

if __name__ == '__main__':
    main()
