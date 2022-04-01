"""
TRAIN ON X, EVALUATE ON X (using top_14_features)
lgbm_01_30 val score       :     0.7870499345743109
lgbm_01_30 submission score:     0.7397 

lgbm_02_02 val score       :     0.7879785572580305
lgbm_02_02 submission score:     0.7407 

lgbm_02_08 val score       :     0.7849279166234973
lgbm_02_08 submission score:     0.7335 
"""
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y, shuffle=True
)

"""
TRAIN ON X, EVALUATE ON X_VAL (using top_14_features)
lgbm_01_30 val score (on X_val): 0.7890673147362179
lgbm_01_30 submission score:     0.7397 

lgbm_02_02 val score (on X_val): 0.7891370829893372
lgbm_02_02 submission score:     0.7407 

lgbm_02_08 val score (on X_val): 0.785823090966174
lgbm_02_08 submission score:     0.7335 
"""

# This only works if used with pd.get_dummies() (see last element in list)
top_14_features = [
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id",
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "count_families",
    "other_floor_type_q",
]
