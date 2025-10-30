import pickle
import pandas as pd

print("--- Loading preprocessor.pkl to check features ---")

try:
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    
    # Get the feature names FROM THE FILE
    cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
    num_features = preprocessor.named_transformers_['num'].feature_names_in_
    
    print("\n--- SUCCESS! Found these features: ---")
    
    print("\nNumerical Features:")
    print(list(num_features))
    
    print("\nCategorical Features:")
    print(list(cat_features))
    
    print("\n--- Please copy and paste all the text above (including the lists) back to me. ---")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(e)
    print("Please paste this error message back to me.")