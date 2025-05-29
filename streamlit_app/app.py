#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import logging
import uuid
import re # For parse_age_years

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llama_cpp, fallback gracefully if not available
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    logger.warning("llama_cpp not available. LLM explanations will be disabled.")
    LLAMA_AVAILABLE = False
    Llama = None

# --- Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

# --- Application Constants & Configuration ---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REC_MODEL_PATH = os.path.join(SCRIPT_DIR, "pet_model.h5") # Make it absolute
PETS_DATA_PATH = os.path.join(SCRIPT_DIR, "data/pets_silver_local/pets_silver") # Make it absolute
ADOPTERS_DATA_PATH = os.path.join(SCRIPT_DIR, "data/adopters_local/Adopters") # Make it absolute
TOP_K_RECOMMENDATIONS = 3

# --- Llama Model Configuration ---
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", os.path.join(SCRIPT_DIR, "models/llama-3-8b-instruct.Q4_K_M.gguf")) # Make it absolute
N_GPU_LAYERS = -1 
N_CTX = 2048

# --- Feature Definitions (must match 03_train_wd_model.py) ---
# These are the features the recommendation model expects.
# The names in `adopter_profile_from_ui` and `pet_row_from_df` (see preprocess_input_for_model)
# must be mapped to these expected feature names.
MODEL_NUMERICAL_FEATURES = [
    'age',                     # Was 'adopter_age'
    'household_size',          # Was 'adopter_household_size'
    'pet_age_years'            # Stays as is, was previously commented out in preprocess
]
MODEL_CATEGORICAL_FEATURES = [
    'housing',                 # Was 'adopter_housing_type'
    'activity_level',          # Was 'adopter_activity_level'
    'has_owned_pets',          # Was 'adopter_has_prior_pets'
    'pet_type',                # Was 'pet_animal_type'
    'pet_breed',               # Stays as is
    'pet_color',               # Stays as is
    'pet_size',                # Stays as is, was previously commented out in preprocess
    'pet_activity_needs',      # Stays as is, was previously commented out in preprocess
    'pet_needs_experienced_owner', # Stays as is, was previously commented out in preprocess
    'pet_good_with_children'   # Stays as is, was previously commented out in preprocess
]

# Constants for parsing pet age
DEFAULT_PET_AGE_YEARS = 1.0 # From PairGCPLoad.py

def parse_age_years(age_str: str) -> float:
    """Parse age string (e.g., '2 years' or '6 months') into years."""
    if pd.isna(age_str) or not isinstance(age_str, str): # Added type check
        return DEFAULT_PET_AGE_YEARS
        
    match = re.match(r'(\d+)\s+(year|month)s?', age_str.lower())
    if not match:
        # Attempt to parse just a number as years if no unit
        match_num_only = re.match(r'(\d+)', age_str.lower())
        if match_num_only:
            return float(match_num_only.group(1))
        return DEFAULT_PET_AGE_YEARS # Default if can't parse
        
    number, unit = match.groups()
    if unit == 'year':
        return float(number)
    else:  # months
        return float(number) / 12.0

# --- Initialize Llama Model ---
@st.cache_resource
def load_llama_model(model_path: str):
    """Load Llama model for generating explanations."""
    if not LLAMA_AVAILABLE:
        logger.warning("Llama not available, skipping model load.")
        return None
        
    if not os.path.exists(model_path):
        logger.warning(f"Llama model not found at {model_path}. LLM explanations will not be available.")
        return None
    try:
        logger.info(f"Loading Llama model from {model_path}")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False
        )
        logger.info("Llama model loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to load Llama model from {model_path}: {e}")
        return None

llm_model = load_llama_model(LLAMA_MODEL_PATH)

# --- Load Recommendation Model & Data ---
@st.cache_resource
def load_rec_model(model_path: str):
    logger.info(f"Loading recommendation model from {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Recommendation model file not found: {model_path}")
        return None
    try:
        # Load the model without custom objects first
        model = tf.keras.models.load_model(model_path, compile=False) # Set compile=False initially
        # If you need to compile (e.g., for further training or specific metrics access):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Example, use actual if known
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # Example, use actual if known
        )
        logger.info("Recommendation model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading recommendation model: {e}")
        # Provide more specific error message if it's about custom objects
        if "Unknown layer" in str(e) or "Unknown metric" in str(e):
            st.error(f"Error loading recommendation model: {e}. This might be due to custom layers/objects not being recognized. Ensure the model was saved correctly or provide custom_objects to load_model.")
        else:
            st.error(f"Error loading recommendation model from {model_path}: {e}")
        return None

@st.cache_data
def load_parquet_data(data_path: str, data_name: str) -> pd.DataFrame:
    logger.info(f"Loading {data_name} data from {data_path}")
    # Check if the path is a directory (for partitioned parquet) or a file
    if not os.path.exists(data_path):
        logger.error(f"{data_name} data path does not exist: {data_path}")
        st.error(f"{data_name} data path does not exist: {data_path}. Please check the path.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"{data_name} data loaded successfully from {data_path}. Shape: {df.shape}. Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading {data_name} data from {data_path}: {e}")
        st.error(f"Error loading {data_name} data from {data_path}: {e}")
        return pd.DataFrame()

rec_model_loaded = load_rec_model(REC_MODEL_PATH)
pets_df = load_parquet_data(PETS_DATA_PATH, "Pets")
# Adopters data for UI options. Column names here should match the source file.
adopters_df_for_options = load_parquet_data(ADOPTERS_DATA_PATH, "Adopters (for UI options)")


# --- Feature Preprocessing Function for Keras Model ---
def preprocess_input_for_model(adopter_profile_from_ui: dict, pet_row_from_df: pd.Series) -> dict:
    """
    Prepare input dictionary for the Keras model prediction.
    - adopter_profile_from_ui: A dictionary containing adopter's choices from Streamlit UI.
                               Keys should be simple (e.g., 'age', 'housing', 'activity').
    - pet_row_from_df: A pandas Series representing a single pet from the pets_df.
                       Column names should match those in the pets_silver dataset.
    Returns:
        A dictionary where keys are feature names expected by the model
        (e.g., 'adopter_age', 'pet_age_years') and values are tf.constant tensors
        of shape (1, 1).
    """
    input_dict = {}

    # --- Map Adopter UI input to MODEL_NUMERICAL_FEATURES ---
    # Keys used here MUST match the Keras model's expected input feature names.
    input_dict['age'] = np.array([[float(adopter_profile_from_ui['age'])]], dtype=np.float32)
    input_dict['household_size'] = np.array([[int(adopter_profile_from_ui['household_size'])]], dtype=np.float32)

    # --- Derive and Map Pet data to MODEL_NUMERICAL_FEATURES ---
    pet_age_str = pet_row_from_df.get('Age upon Outcome', None)
    input_dict['pet_age_years'] = np.array([[parse_age_years(pet_age_str)]], dtype=np.float32)


    # --- Map Adopter UI input to MODEL_CATEGORICAL_FEATURES ---
    # Keys used here MUST match the Keras model's expected input feature names.
    input_dict['housing'] = np.array([[str(adopter_profile_from_ui['housing'])]], dtype=object)
    input_dict['activity_level'] = np.array([[str(adopter_profile_from_ui['activity'])]], dtype=object)
    has_pets_str = str(adopter_profile_from_ui['has_prior_pets']) # UI sends True/False, model needs string
    input_dict['has_owned_pets'] = np.array([[has_pets_str]], dtype=object)
    
    # --- Map Pet data to MODEL_CATEGORICAL_FEATURES ---
    # Keys used here MUST match the Keras model's expected input feature names.
    input_dict['pet_type'] = np.array([[str(pet_row_from_df.get('Animal Type', 'Unknown'))]], dtype=object)
    input_dict['pet_breed'] = np.array([[str(pet_row_from_df.get('Breed', 'Unknown'))]], dtype=object)
    input_dict['pet_color'] = np.array([[str(pet_row_from_df.get('Color', 'Unknown'))]], dtype=object)
    
    # Restore previously commented-out pet features, ensuring keys match updated MODEL_CATEGORICAL_FEATURES
    input_dict['pet_size'] = np.array([[str(pet_row_from_df.get('Size', 'Medium'))]], dtype=object) 
    input_dict['pet_activity_needs'] = np.array([[str(pet_row_from_df.get('activity_needs', 'Moderate'))]], dtype=object)
    input_dict['pet_needs_experienced_owner'] = np.array([[str(pet_row_from_df.get('needs_experienced_owner', False))]], dtype=object)
    input_dict['pet_good_with_children'] = np.array([[str(pet_row_from_df.get('good_with_children', True))]], dtype=object)

    # Sanity check: all model features (from updated global lists) must be present
    all_expected_features = MODEL_NUMERICAL_FEATURES + MODEL_CATEGORICAL_FEATURES
    
    for feature_name in all_expected_features:
        if feature_name not in input_dict:
            raise KeyError(f"Feature '{feature_name}' (expected by model based on global lists) is missing from the input_dict. "
                           f"Adopter Profile Keys: {list(adopter_profile_from_ui.keys())}, "
                           f"Pet Row Columns: {pet_row_from_df.index.tolist()}")
        # Ensure correct shape (1,1) and type for Keras functional model inputs
        current_val = input_dict[feature_name]
        if not isinstance(current_val, np.ndarray) or current_val.shape != (1,1):
             logger.warning(f"Feature {feature_name} has unexpected shape/type: {type(current_val)}, {current_val.shape if hasattr(current_val, 'shape') else 'N/A'}. Attempting reshape.")
             try:
                 # Check if feature is numerical or categorical based on our definitive lists
                 if feature_name in MODEL_NUMERICAL_FEATURES:
                     input_dict[feature_name] = np.array([[current_val.item() if hasattr(current_val, 'item') else current_val]], dtype=np.float32)
                 else: # Categorical
                     input_dict[feature_name] = np.array([[str(current_val.item() if hasattr(current_val, 'item') else current_val)]], dtype=object)
             except Exception as e:
                 logger.error(f"Failed to reshape/recast {feature_name}: {e}")
                 raise ValueError(f"Could not prepare feature {feature_name} correctly for the model.") from e

    # Convert numpy arrays to tf.constant - Keras model.predict expects this structure
    # when inputs are a dictionary.
    tf_input_dict = {name: tf.constant(val) for name, val in input_dict.items()}
    
    # No longer filter based on a smaller list of error_features. Pass all prepared features.
    # dropped_keys logic removed as we expect all keys in input_dict to be used.
    
    # Final check if all expected keys (from global lists) are present in the final dictionary
    for expected_key in all_expected_features:
        if expected_key not in tf_input_dict:
            raise KeyError(f"Critical error: Expected feature '{expected_key}' not found in final_tf_input_dict. Available keys: {list(tf_input_dict.keys())}")

    return tf_input_dict

# --- Function to generate explanation (Original from line 112) ---
@st.cache_data 
def get_match_explanation(_adopter_profile: dict, _pet_details: pd.Series, _llm_model: Llama, pet_id_for_cache_key: str): # Added pet_id_for_cache_key and changed _adopter_profile type
    if not _llm_model:
        logger.warning("Llama model not available. Skipping explanation.")
        return "LLM model not available. Cannot generate explanation."
    
    logger.info(f"Generating explanation for Adopter: {_adopter_profile.get('profile_uuid', 'Unknown AdopterID')}, Pet: {pet_id_for_cache_key} (cache key)")

    # Using .get for safety from _adopter_profile (dict from UI)
    adopter_age = _adopter_profile.get('age', 'N/A')
    adopter_hh_size = _adopter_profile.get('household_size', 'N/A')
    adopter_housing = _adopter_profile.get('housing', 'N/A') # Key used in UI form
    adopter_activity = _adopter_profile.get('activity', 'N/A') # Key used in UI form
    adopter_has_prior_pet_bool = _adopter_profile.get('has_prior_pets', False) # Key used in UI form
    adopter_has_prior_pet_text = "Yes" if adopter_has_prior_pet_bool else "No"
    
    # Using .get for safety from _pet_details (Series from pets_df)
    pet_name = _pet_details.get('Name', 'This pet')
    pet_animal_type = _pet_details.get('Animal Type', 'N/A')
    pet_breed = _pet_details.get('Breed', 'N/A')
    pet_color = _pet_details.get('Color', 'N/A')
    pet_age_outcome = _pet_details.get('Age upon Outcome', 'N/A')
    pet_sex = _pet_details.get('Sex upon Outcome', 'N/A')
    pet_intake_condition = _pet_details.get('Intake Condition', 'N/A')
    # New features that might be useful for explanation, if available in pets_df:
    pet_size_exp = _pet_details.get('Size', 'unknown size')
    pet_activity_exp = _pet_details.get('activity_needs', 'unknown activity needs')
    pet_exp_owner_exp = "needs an experienced owner" if _pet_details.get('needs_experienced_owner', False) else "doesn't necessarily need an experienced owner"
    pet_good_children_exp = "is good with children" if _pet_details.get('good_with_children', True) else "may not be suitable for a home with children"


    system_prompt = "You are a helpful assistant at an animal shelter, skilled at explaining why a pet and an adopter could be a good match. Be positive, encouraging, and concise (2-3 sentences)."
    
    # Plain multiline string for the template part, then format it.
    user_prompt_lines = [
        "A potential adopter has the following profile:",
        f"- Age: {adopter_age}",
        f"- Household Size: {adopter_hh_size}",
        f"- Housing Type: {adopter_housing}",
        f"- Activity Level: {adopter_activity}",
        f"- Has Owned Pets Before: {adopter_has_prior_pet_text}",
        "\nThey are being matched with a pet with the following details:",
        f"- Name: {pet_name}",
        f"- Type: {pet_animal_type}",
        f"- Breed: {pet_breed}",
        f"- Color: {pet_color}",
        f"- Age upon Outcome: {pet_age_outcome} (Size: {pet_size_exp})",
        f"- Sex upon Outcome: {pet_sex}",
        f"- Intake Condition: {pet_intake_condition}",
        f"- Temperament notes: This pet {pet_activity_exp}, {pet_exp_owner_exp}, and {pet_good_children_exp}.",
        "\nBased on this information, briefly explain in 2-3 concise sentences why this pet might be a good match for this adopter.",
        "Focus on compatibility aspects derived from their profiles."
    ]
    user_prompt_template = "\n".join(user_prompt_lines)
    
    full_prompt_for_completion = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    try:
        logger.info(f"Sending prompt to Llama model for pet {_pet_details.get('Animal ID', 'Unknown ID')}:\n{user_prompt_template[:300]}...")

        completion = _llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        explanation = completion['choices'][0]['message']['content'].strip()
        
        logger.info(f"Generated Llama explanation for adopter profile and pet {_pet_details.get('Animal ID', 'Unknown ID')}")
        return explanation
    except Exception as e:
        # Log the full prompt if an error occurs during generation
        logger.error(f"Error generating Llama explanation for pet {_pet_details.get('Animal ID', 'Unknown ID')}: {e}\nFull prompt was:\n{full_prompt_for_completion}")
        return "Could not generate an explanation using the local LLM at this time."

# --- Recommendation Logic (Original from line ~165) ---
@st.cache_data 
def rank_pets(adopter_profile: dict, all_pets_df: pd.DataFrame, _model: tf.keras.Model, top_k: int = TOP_K_RECOMMENDATIONS):
    # adopter_profile is now a dict from UI session state
    adopter_identifier = adopter_profile.get('profile_uuid', str(uuid.uuid4())) 
    logger.info(f"Ranking pets for adopter profile ID: {adopter_identifier}")
    scores = []

    if not _model:
        logger.error("Recommendation Model is not loaded. Cannot make predictions.")
        st.error("Recommendation Model is not loaded. Cannot make predictions.")
        return pd.DataFrame(), [] # Return empty dataframe and empty list of scored pets
    
    if all_pets_df.empty:
        logger.warning("Pets DataFrame is empty. Cannot rank pets.")
        st.warning("Pet data is not available to rank.")
        return pd.DataFrame(), []

    # Sample pets to score for performance in interactive app
    # Ensure sampling doesn't fail if len(all_pets_df) < 2000
    sample_n = min(len(all_pets_df), 2000)
    if sample_n == 0:
        logger.warning("No pets to sample for scoring.")
        return pd.DataFrame(), []
        
    pets_to_score_df = all_pets_df.sample(sample_n, random_state=42) 
    
    scored_pet_details = [] # To store rows of pets that were successfully scored

    for _, pet_row in pets_to_score_df.iterrows():
        try:
            # `adopter_profile` is the dict from UI, `pet_row` is a Series from pets_df
            model_input = preprocess_input_for_model(adopter_profile, pet_row)
            score = _model.predict(model_input, verbose=0)[0][0]
            scores.append({'pet_id': pet_row['Animal ID'], 'score': score})
            scored_pet_details.append(pet_row) # Save the full pet_row
        except Exception as e:
            # Log the adopter profile keys and pet_row columns for easier debugging
            logger.error(f"Error predicting for pet {pet_row.get('Animal ID', 'Unknown ID')} with adopter {adopter_identifier}: {e}. "
                         f"Adopter Profile Keys: {list(adopter_profile.keys())}, "
                         f"Pet Row Columns: {pet_row.index.tolist()}")
            # Log problematic model_input only if essential and careful about verbosity/PII
            # logger.debug(f"Problematic model_input: {model_input}") 
            continue 

    if not scores:
        logger.info(f"No scores were generated for adopter profile ID: {adopter_identifier}")
        # st.info("Could not generate scores for any pets with the current profile.") # Already handled by empty df
        return pd.DataFrame(), []

    ranked_scores_df = pd.DataFrame(scores).sort_values(by='score', ascending=False)
    
    # Merge with (already sampled and scored) pet details to get full pet info
    # `scored_pet_details` is a list of Series, convert to DataFrame
    scored_pets_info_df = pd.DataFrame(scored_pet_details)
    
    # We need to ensure 'pet_id' in ranked_scores_df matches 'Animal ID' in scored_pets_info_df for merging
    # If 'Animal ID' is already the index of scored_pets_info_df, adjust merge
    # For simplicity, assuming 'Animal ID' is a column
    if 'Animal ID' not in scored_pets_info_df.columns:
        logger.error("'Animal ID' not found in columns of scored_pets_info_df. Cannot merge for full details.")
        # Fallback: return ranked scores but pet details might be just IDs
        # Or, try to find another ID. For now, we rely on 'Animal ID'.
        # This indicates an issue with `scored_pet_details` assembly or `pets_df` structure.
        # We can reconstruct from pets_df using the pet_ids in ranked_scores_df.
        top_k_pet_ids = ranked_scores_df.head(top_k)['pet_id'].tolist()
        final_ranked_df = all_pets_df[all_pets_df['Animal ID'].isin(top_k_pet_ids)].copy()
        # Need to add the 'score' back to this final_ranked_df
        final_ranked_df = pd.merge(final_ranked_df, ranked_scores_df[['pet_id', 'score']], left_on='Animal ID', right_on='pet_id')
        final_ranked_df = final_ranked_df.sort_values(by='score', ascending=False).drop_duplicates(subset=['Animal ID'])

    else:
        # Merge ranked scores with the details of the pets that were scored
        ranked_df_full_details = pd.merge(
            ranked_scores_df,
            scored_pets_info_df, # This contains the subset of pets that were scored
            left_on='pet_id',
            right_on='Animal ID', # Assuming 'Animal ID' is the common key
            how='left' 
        ).drop_duplicates(subset=['pet_id']) # Should not be necessary if IDs are unique in pets_to_score_df

    logger.info(f"Ranked {len(ranked_scores_df)} pets for adopter {adopter_identifier}. Top {top_k} scores: {ranked_scores_df.head(top_k)['score'].tolist()}")
    # Return the top_k from the dataframe that has full details
    if 'final_ranked_df' in locals():
         return final_ranked_df.head(top_k), ranked_scores_df # Return all scores for potential debugging/analysis
    return ranked_df_full_details.head(top_k), ranked_scores_df


# --- UI Layout ---
st.title("🐾 Pet-Match Recommender (LLM Enhanced)")

# Initialize session state for adopter profile and generated UUID
if 'adopter_profile' not in st.session_state:
    st.session_state.adopter_profile = {}
if 'profile_uuid' not in st.session_state:
    st.session_state.profile_uuid = str(uuid.uuid4())
if 'recommendations_df' not in st.session_state:
    st.session_state.recommendations_df = pd.DataFrame()
if 'all_scores_df' not in st.session_state: # To store all scores from rank_pets
    st.session_state.all_scores_df = pd.DataFrame()


# --- Model and Data Loading Checks ---
models_loaded_correctly = True
if llm_model is None:
    st.warning("Local LLM (Llama) could not be initialized. LLM explanations will not be available. Please check logs and model path.")
    # models_loaded_correctly = False # Explanations are optional

if rec_model_loaded is None:
    st.error("Recommendation model could not be loaded. Please check logs and ensure `pet_model.h5` is correctly placed.")
    models_loaded_correctly = False
if pets_df.empty:
    st.error("Pet data (`pets_silver_local/pets_silver`) is not loaded or is empty. Please check the data path.")
    models_loaded_correctly = False
if adopters_df_for_options.empty:
    st.warning("Adopter options data (`adopters_local/Adopters`) is not loaded. Dropdowns might use defaults or be empty.")
    # Not strictly critical for operation if defaults are handled, but good to warn.


if models_loaded_correctly:
    st.sidebar.header("Define Your Adopter Profile:")
    
    # --- Adopter Profile Input Fields ---
    # Use keys that are descriptive and match what `preprocess_input_for_model` expects 
    # from `adopter_profile_from_ui` dictionary.
    
    # Basic Info
    age = st.sidebar.number_input("Your Age:", min_value=18, max_value=100, value=st.session_state.adopter_profile.get('age', 30), key="ui_age")
    household_size = st.sidebar.number_input("Household Size (number of people):", min_value=1, max_value=10, value=st.session_state.adopter_profile.get('household_size', 2), key="ui_household_size")

    # Housing & Lifestyle
    # Dynamically populate options from adopters_df_for_options if available, else use defaults
    default_housing_options = ['House', 'Apartment', 'Condo', 'Townhouse', 'Other']
    housing_options = adopters_df_for_options['housing'].unique().tolist() if 'housing' in adopters_df_for_options.columns and not adopters_df_for_options['housing'].empty else default_housing_options
    housing_current_val = st.session_state.adopter_profile.get('housing', housing_options[0] if housing_options else default_housing_options[0])
    housing_idx = housing_options.index(housing_current_val) if housing_current_val in housing_options else 0
    housing = st.sidebar.selectbox("Housing Type:", options=housing_options, index=housing_idx, key="ui_housing")

    default_activity_options = ['Low', 'Moderate', 'High', 'Active', 'Very Active']
    activity_options = adopters_df_for_options['activity'].unique().tolist() if 'activity' in adopters_df_for_options.columns and not adopters_df_for_options['activity'].empty else default_activity_options
    activity_current_val = st.session_state.adopter_profile.get('activity', activity_options[0] if activity_options else default_activity_options[0])
    activity_idx = activity_options.index(activity_current_val) if activity_current_val in activity_options else 0
    activity = st.sidebar.selectbox("Your Activity Level:", options=activity_options, index=activity_idx, key="ui_activity")

    # Pet Experience
    # The model expects string 'True' or 'False' for 'adopter_has_prior_pets'
    # UI gives True/False boolean, convert in preprocess_input_for_model
    has_prior_pets_radio_options = {'Yes': True, 'No': False}
    has_prior_pets_current_bool = st.session_state.adopter_profile.get('has_prior_pets', False) # Default to False (boolean)
    # Find the radio option string ('Yes' or 'No') corresponding to the boolean
    current_radio_selection = 'No' # Default for radio display
    for k,v in has_prior_pets_radio_options.items():
        if v == has_prior_pets_current_bool:
            current_radio_selection = k
            break
    
    has_prior_pets_selection_str = st.sidebar.radio( # String 'Yes' or 'No'
        "Have you owned pets before?",
        options=list(has_prior_pets_radio_options.keys()),
        index=list(has_prior_pets_radio_options.keys()).index(current_radio_selection),
        key="ui_has_prior_pets"
    )
    # Convert radio selection string back to boolean for storing in session_state,
    # but preprocess_input_for_model will convert it to string 'True'/'False' for the model.
    has_prior_pets_bool = has_prior_pets_radio_options[has_prior_pets_selection_str]


    if st.sidebar.button("✨ Find My Pet Matches ✨", key="find_matches_button"):
        # Store current UI selections into session_state.adopter_profile
        # This `adopter_profile` dict is what's passed to `rank_pets` and then `preprocess_input_for_model`
        st.session_state.adopter_profile = {
            'profile_uuid': st.session_state.profile_uuid, # Persist UUID
            'age': age,
            'household_size': household_size,
            'housing': housing, # This is 'House', 'Apartment' etc.
            'activity': activity, # This is 'Low', 'Moderate', 'High'
            'has_prior_pets': has_prior_pets_bool, # This is True/False boolean
            # Add any other profile fields needed by the model or explanation here
        }
        
        # Call ranking function
        # `st.session_state.adopter_profile` is the dict for `adopter_profile` argument in `rank_pets`
        # `pets_df` is the DataFrame of all pets
        # `rec_model_loaded` is the Keras model
        with st.spinner("Searching for your perfect match... 🐕🐈"):
            recommendations, all_scores = rank_pets(
                st.session_state.adopter_profile, 
                pets_df, 
                rec_model_loaded, 
                TOP_K_RECOMMENDATIONS
            )
            st.session_state.recommendations_df = recommendations
            st.session_state.all_scores_df = all_scores # Save all scores for inspection

    # --- Display Recommendations ---
    if not st.session_state.recommendations_df.empty:
        st.subheader("🌟 Here are your top pet recommendations: 🌟")
        
        # Create columns for layout if more than 1 recommendation
        num_recs = len(st.session_state.recommendations_df)
        cols = st.columns(num_recs) if num_recs > 0 else []

        for i, (_, pet_rec_series) in enumerate(st.session_state.recommendations_df.iterrows()):
            # pet_rec_series is a row from the ranked recommendations DataFrame
            # It should contain all necessary pet details from the original pets_df plus the 'score'
            
            target_col = cols[i % len(cols)] if cols else st # Use columns or st directly
            
            with target_col:
                st.markdown(f"### {pet_rec_series.get('Name', 'Pet ID: ' + str(pet_rec_series.get('Animal ID', 'N/A')))}")
                
                # Display key pet info
                # Ensure these columns exist in `st.session_state.recommendations_df`
                # (which comes from merging rank_pets output with pets_df columns)
                pet_info_md = f"""
                - **ID:** {pet_rec_series.get('Animal ID', 'N/A')}
                - **Type:** {pet_rec_series.get('Animal Type', 'N/A')}
                - **Breed:** {pet_rec_series.get('Breed', 'N/A')}
                - **Age:** {pet_rec_series.get('Age upon Outcome', 'N/A')}
                - **Size:** {pet_rec_series.get('Size', 'N/A')}
                - **Sex:** {pet_rec_series.get('Sex upon Outcome', 'N/A')}
                - **Color:** {pet_rec_series.get('Color', 'N/A')}
                """
                st.markdown(pet_info_md)
                st.metric(label="Match Score", value=f"{pet_rec_series.get('score', 0.0):.2%}")

                # LLM Explanation
                if llm_model:
                    with st.spinner("Generating match explanation..."):
                        # Pass the same adopter_profile dict from session_state
                        # Pass the pet_rec_series (which is a row from pets_df with score)
                        explanation = get_match_explanation(
                            st.session_state.adopter_profile, 
                            pet_rec_series, 
                            llm_model,
                            pet_rec_series.get('Animal ID', f"pet_idx_{i}") # Pass Animal ID for cache key
                        )
                        st.info(f"**Why this could be a great match:**\n{explanation}")
                else:
                    st.markdown("_LLM explanations are currently unavailable._")
                st.divider()

    elif st.session_state.get('find_matches_button_clicked', False): # Check if button was clicked
        st.info("No recommendations found based on your profile. Try adjusting your preferences!")

    # For debugging: show all scores if available
    if not st.session_state.all_scores_df.empty and os.getenv("STREAMLIT_DEBUG_MODE") == "true":
        st.sidebar.subheader("Debug: All Pet Scores (Sampled)")
        st.sidebar.dataframe(st.session_state.all_scores_df)
        st.sidebar.caption("Note: Scores are for a sample of pets for performance.")

else:
    st.error("Application cannot start due to loading errors. Please check the console logs for details.")

# Persist button click state to avoid re-showing "No recommendations" on first load
if 'find_matches_button' in st.session_state and st.session_state.find_matches_button:
    st.session_state.find_matches_button_clicked = True
else:
    st.session_state.find_matches_button_clicked = False
