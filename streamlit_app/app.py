#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import logging
import uuid
import re
from typing import Dict, Tuple, Optional, List
from contextlib import contextmanager
import openai # Import OpenAI

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Pet Match Recommender",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Application Constants ---
class Config:
    """Application configuration constants."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Paths
    REC_MODEL_PATH = os.path.join(SCRIPT_DIR, "pet_model.h5")
    PETS_DATA_PATH = os.path.join(PROJECT_ROOT, "data/pets_silver_local/pets_silver")
    ADOPTERS_DATA_PATH = os.path.join(PROJECT_ROOT, "data/adopters_local/Adopters")
    
    # Model parameters
    TOP_K_RECOMMENDATIONS = 3
    BATCH_SIZE = 2000
    DEFAULT_PET_AGE_YEARS = 1.0
    
    # OpenAI parameters
    OPENAI_MODEL = "gpt-3.5-turbo" # Or your preferred model
    
    # Feature definitions (must match training script)
    NUMERICAL_FEATURES = ['age', 'household_size', 'pet_age_years']
    CATEGORICAL_FEATURES = [
        'housing', 'activity_level', 'has_owned_pets', 'pet_type', 
        'pet_breed', 'pet_color', 'pet_size', 'pet_activity_needs',
        'pet_needs_experienced_owner', 'pet_good_with_children'
    ]

# --- Error Handling Utilities ---
@contextmanager
def error_handler(operation_name: str, show_user_error: bool = True):
    """Context manager for consistent error handling."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}", exc_info=True)
        if show_user_error:
            st.error(f"Error in {operation_name}: {str(e)}")

def safe_operation(func, default_value=None, operation_name: str = "operation"):
    """Safely execute an operation with error handling."""
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        return default_value

# --- Utility Functions ---
def parse_age_years(age_str: str) -> float:
    """Parse age string into years with improved error handling."""
    if pd.isna(age_str) or not isinstance(age_str, str):
        return Config.DEFAULT_PET_AGE_YEARS
    
    try:
        # Try specific pattern first
        match = re.match(r'(\d+)\s+(year|month)s?', age_str.lower().strip())
        if match:
            number, unit = match.groups()
            return float(number) if unit == 'year' else float(number) / 12.0
        
        # Try just number
        match_num = re.match(r'(\d+)', age_str.lower().strip())
        if match_num:
            return float(match_num.group(1))
            
    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not parse age '{age_str}': {e}")
    
    return Config.DEFAULT_PET_AGE_YEARS

# --- OpenAI Client Setup ---
@st.cache_resource
def get_openai_client() -> Optional[openai.OpenAI]:
    """Initialize and return the OpenAI client. API key is fetched from st.secrets."""
    
    # Explicitly log the type and content of what st.secrets retrieves
    raw_api_key_from_secrets = st.secrets.get("OPENAI_API_KEY")
    logger.info(f"Attempting to retrieve OPENAI_API_KEY from st.secrets.")
    logger.info(f"  Type of retrieved key: {type(raw_api_key_from_secrets)}")
    logger.info(f"  Value of retrieved key (first 5 chars if string): '{(str(raw_api_key_from_secrets)[:5] + "...") if isinstance(raw_api_key_from_secrets, str) and len(raw_api_key_from_secrets) > 5 else raw_api_key_from_secrets}'")

    api_key = raw_api_key_from_secrets
    # Treat empty string as not found as well
    if not api_key or not isinstance(api_key, str) or api_key.strip() == "": 
        logger.warning("OpenAI API key not found or is empty in st.secrets. LLM explanations will be disabled.")
        st.warning("OpenAI API key not configured or is empty. Explanations are disabled.", icon="🔑")
        return None
    try:
        client = openai.OpenAI(api_key=api_key.strip()) # Ensure to strip whitespace
        logger.info("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        st.error("Failed to initialize OpenAI client. Explanations may be unavailable.")
        return None

# --- Model and Data Loading ---
@st.cache_resource
def load_recommendation_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load TensorFlow recommendation model with error handling."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    with error_handler("recommendation model loading"):
        logger.info(f"Loading recommendation model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Compile with basic settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Recommendation model loaded successfully.")
        return model
    
    return None

@st.cache_data
def load_parquet_data(data_path: str, data_name: str) -> pd.DataFrame:
    """Load parquet data with comprehensive error handling."""
    if not os.path.exists(data_path):
        logger.error(f"{data_name} data path does not exist: {data_path}")
        st.error(f"{data_name} data not found. Please check the data path.")
        return pd.DataFrame()
    
    with error_handler(f"{data_name} data loading"):
        logger.info(f"Loading {data_name} data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"{data_name} loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    return pd.DataFrame()

# --- Batch Processing Functions ---
class BatchProcessor:
    """Handles batch processing of pet data for model prediction."""
    
    @staticmethod
    def get_column_safe(df: pd.DataFrame, column_name: str, default_value, batch_size: int) -> pd.Series:
        """Safely extract column with default values."""
        if column_name in df.columns:
            # Determine the value to use for filling NaNs within an existing column
            fill_val_for_existing_column_nans = default_value
            if default_value is None:
                # If the intended default for Nones is None, use an empty string for fillna.
                # parse_age_years will handle this and convert to DEFAULT_PET_AGE_YEARS.
                # For other types, None might be a valid fill, but for age string parsing, this is safer.
                fill_val_for_existing_column_nans = ""
            
            series = df[column_name].fillna(value=fill_val_for_existing_column_nans)
            if len(series) != batch_size:
                # This check might be too noisy if pets_df is sometimes shorter than batch_size (e.g. during sampling)
                # Consider if this warning is essential or can be logged at a lower level / removed if sampling handles size.
                logger.debug(f"Column {column_name} length after fillna: {len(series)}, expected batch_size: {batch_size}. This might be OK if df was sampled smaller than batch_size.")
            return series
        else:
            # If column doesn't exist, create a new series filled with the original default_value.
            # pd.Series([None] * batch_size) is valid if default_value is None.
            logger.warning(f"Column {column_name} not found, creating series with default: {default_value}")
            return pd.Series([default_value] * batch_size)
    
    @staticmethod
    def validate_batch_features(batch_input: Dict[str, np.ndarray], expected_features: List[str], batch_size: int) -> bool:
        """Validate that all expected features are present with correct shapes."""
        for feature_name in expected_features:
            if feature_name not in batch_input:
                logger.error(f"Missing feature: {feature_name}")
                return False
            
            expected_shape = (batch_size, 1)
            actual_shape = batch_input[feature_name].shape
            if actual_shape != expected_shape:
                logger.error(f"Feature {feature_name} shape mismatch: {actual_shape} vs {expected_shape}")
                return False
        
        return True
    
    @classmethod
    def preprocess_batch(cls, adopter_profile: Dict, pets_df: pd.DataFrame) -> Optional[Dict[str, tf.Tensor]]:
        """Preprocess a batch of pets for model prediction."""
        batch_size = len(pets_df)
        if batch_size == 0:
            logger.warning("Empty batch provided for preprocessing")
            return None
        
        try:
            batch_input = {}
            
            # Adopter features (repeated for batch)
            batch_input['age'] = np.full((batch_size, 1), float(adopter_profile['age']), dtype=np.float32)
            batch_input['household_size'] = np.full((batch_size, 1), float(adopter_profile['household_size']), dtype=np.float32)
            batch_input['housing'] = np.full((batch_size, 1), str(adopter_profile['housing']), dtype=object)
            batch_input['activity_level'] = np.full((batch_size, 1), str(adopter_profile['activity']), dtype=object)
            batch_input['has_owned_pets'] = np.full((batch_size, 1), str(adopter_profile['has_prior_pets']), dtype=object)
            
            # Pet features (vectorized processing)
            age_series = cls.get_column_safe(pets_df, 'Age upon Outcome', None, batch_size)
            pet_ages = np.array([parse_age_years(age) for age in age_series], dtype=np.float32).reshape(-1, 1)
            batch_input['pet_age_years'] = pet_ages
            
            # Categorical pet features
            pet_features = {
                'pet_type': ('Animal Type', 'Unknown'),
                'pet_breed': ('Breed', 'Unknown'),
                'pet_color': ('Color', 'Unknown'),
                'pet_size': ('Size', 'Medium'),
                'pet_activity_needs': ('activity_needs', 'Moderate'),
                'pet_needs_experienced_owner': ('needs_experienced_owner', False),
                'pet_good_with_children': ('good_with_children', True)
            }
            
            for feature_name, (column_name, default_value) in pet_features.items():
                series = cls.get_column_safe(pets_df, column_name, default_value, batch_size)
                batch_input[feature_name] = series.astype(str).values.reshape(-1, 1)
            
            # Validate batch
            all_features = Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES
            if not cls.validate_batch_features(batch_input, all_features, batch_size):
                return None
            
            # Convert to TensorFlow tensors
            tf_batch_input = {name: tf.constant(val) for name, val in batch_input.items()}
            
            logger.info(f"Successfully preprocessed batch of {batch_size} pets")
            return tf_batch_input
            
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}", exc_info=True)
            return None

# --- Recommendation Logic ---
@st.cache_data
def rank_pets_batch(adopter_profile: Dict, all_pets_df: pd.DataFrame, _model: tf.keras.Model, 
                   top_k: int = Config.TOP_K_RECOMMENDATIONS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank pets using batch prediction with comprehensive error handling."""
    adopter_id = adopter_profile.get('profile_uuid', 'unknown')
    logger.info(f"Starting pet ranking for adopter {adopter_id}")
    
    # Validation
    if _model is None:
        logger.error("No model provided for ranking")
        st.error("Recommendation model is not available")
        return pd.DataFrame(), pd.DataFrame()
    
    if all_pets_df.empty:
        logger.warning("No pet data available for ranking")
        st.warning("No pet data available")
        return pd.DataFrame(), pd.DataFrame()
    
    # Sample pets for performance
    sample_size = min(len(all_pets_df), Config.BATCH_SIZE)
    pets_sample = all_pets_df.sample(n=sample_size, random_state=42).copy()
    logger.info(f"Processing {sample_size:,} pets for ranking")
    
    # Batch preprocessing
    with st.spinner("Preparing pet data for analysis..."):
        batch_input = BatchProcessor.preprocess_batch(adopter_profile, pets_sample)
        if batch_input is None:
            logger.error("Batch preprocessing failed")
            st.error("Error processing pet data")
            return pd.DataFrame(), pd.DataFrame()
    
    # Batch prediction
    with st.spinner("Calculating compatibility scores..."):
        try:
            logger.info(f"Making batch prediction for {len(pets_sample)} pets")
            scores = _model.predict(batch_input, verbose=0, batch_size=min(512, len(pets_sample)))
            pets_sample['score'] = scores.flatten()
            logger.info("Batch prediction completed successfully")
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            st.error("Error calculating pet compatibility scores")
            return pd.DataFrame(), pd.DataFrame()
    
    # Sort and return results
    ranked_pets = pets_sample.sort_values('score', ascending=False)
    top_recommendations = ranked_pets.head(top_k)
    
    logger.info(f"Ranking complete. Top {top_k} scores: {top_recommendations['score'].tolist()}")

    # --- Enhanced Logging for Score Verification ---
    # Check for perfect or near-perfect scores
    perfect_threshold = 0.999
    perfect_scores_df = ranked_pets[ranked_pets['score'] >= perfect_threshold]
    num_perfect_scores = len(perfect_scores_df)

    if num_perfect_scores > 0:
        logger.warning(f"{num_perfect_scores} pets achieved a score >= {perfect_threshold} for adopter {adopter_id}.")
        # Log details of up to 3 such pets for inspection
        for i, (_, pet_row) in enumerate(perfect_scores_df.head(3).iterrows()):
            logger.warning(
                f"Perfect Score Pet #{i+1} (ID: {pet_row.get('Animal ID', 'N/A')}, Score: {pet_row['score']:.4f}):\n"
                f"  Pet Type: {pet_row.get('Animal Type')}, Breed: {pet_row.get('Breed')}, Age: {pet_row.get('Age upon Outcome')}, Size: {pet_row.get('Size')}\n"
                f"  Adopter Profile: {adopter_profile}"
            )
    else:
        logger.info(f"No pets achieved a score >= {perfect_threshold} for adopter {adopter_id}.")
    
    # Log score distribution stats
    score_stats = ranked_pets['score'].agg(['count', 'min', 'max', 'mean', 'median', 'std']).to_dict()
    logger.info(f"Score distribution for adopter {adopter_id}: {score_stats}")
    # --- End Enhanced Logging ---

    return top_recommendations, ranked_pets

# --- LLM Explanation Generation (Now OpenAI) ---
@st.cache_data
def generate_match_explanation(_adopter_profile: Dict, _pet_details: pd.Series, 
                             _openai_client: openai.OpenAI, pet_id: str) -> str:
    """Generate LLM explanation using OpenAI API with error handling."""
    if not _openai_client:
        # This case should ideally be handled before calling, 
        # e.g., by checking if the client is available in app_resources
        logger.warning("OpenAI client not available for generating explanation.")
        return "Explanations are currently unavailable (client not initialized)."
    
    try:
        # Extract adopter info safely
        adopter_info = {
            'age': _adopter_profile.get('age', 'N/A'),
            'household_size': _adopter_profile.get('household_size', 'N/A'),
            'housing': _adopter_profile.get('housing', 'N/A'),
            'activity': _adopter_profile.get('activity', 'N/A'),
            'has_prior_pets': 'Yes' if _adopter_profile.get('has_prior_pets', False) else 'No'
        }
        
        # Extract pet info safely
        pet_info = {
            'name': _pet_details.get('Name', 'This pet'),
            'type': _pet_details.get('Animal Type', 'N/A'),
            'breed': _pet_details.get('Breed', 'N/A'),
            'age': _pet_details.get('Age upon Outcome', 'N/A'),
            'size': _pet_details.get('Size', 'unknown size'),
            'sex': _pet_details.get('Sex upon Outcome', 'N/A')
        }
        
        # Create prompt (can be refined for OpenAI if needed)
        system_message = "You are a helpful pet adoption counselor. Be positive, encouraging, and concise (2-3 sentences). Focus on compatibility aspects derived from the provided profiles."
        user_prompt = f"""Consider an adopter who is {adopter_info['age']} years old, has a household of {adopter_info['household_size']}, lives in a {adopter_info['housing']}, has a {adopter_info['activity']} activity level, and {adopter_info['has_prior_pets']} owned pets before. 
They are being matched with a pet named {pet_info['name']}, a {pet_info['age']} {pet_info['type']} ({pet_info['breed']}, {pet_info['size']}).

Why might this pet be a good match for this adopter?"""
        
        logger.info(f"Sending explanation prompt to OpenAI for pet {pet_id}")
        completion = _openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=120, # Adjusted for conciseness
            temperature=0.7
        )
        
        explanation = completion.choices[0].message.content.strip()
        logger.info(f"Generated OpenAI explanation for pet {pet_id}")
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating OpenAI explanation for pet {pet_id}: {e}", exc_info=True)
        # Provide a more user-friendly error if it's an API connection or auth issue
        if isinstance(e, openai.APIConnectionError):
            return "Could not connect to the explanation service. Please try again later."
        if isinstance(e, openai.AuthenticationError):
            return "Explanation service authentication failed. Please check API key configuration."
        return "Could not generate an explanation at this time due to an unexpected error."

# --- Initialize App Resources ---
@st.cache_resource
def initialize_app_resources():
    """Initialize all app resources with error handling."""
    resources = {
        'openai_client': None,
        'rec_model': None,
        'pets_df': pd.DataFrame(),
        'adopters_df': pd.DataFrame(),
        'errors': []
    }
    
    # Initialize OpenAI Client
    resources['openai_client'] = get_openai_client()
    # No explicit error append here, get_openai_client handles logging/st.warning

    # Load recommendation model
    resources['rec_model'] = safe_operation(
        lambda: load_recommendation_model(Config.REC_MODEL_PATH),
        default_value=None,
        operation_name="recommendation model loading"
    )
    if resources['rec_model'] is None:
        resources['errors'].append("Recommendation model failed to load")
    
    # Load data
    resources['pets_df'] = safe_operation(
        lambda: load_parquet_data(Config.PETS_DATA_PATH, "Pets"),
        default_value=pd.DataFrame(),
        operation_name="pets data loading"
    )
    if resources['pets_df'].empty:
        resources['errors'].append("Pet data failed to load")
    
    resources['adopters_df'] = safe_operation(
        lambda: load_parquet_data(Config.ADOPTERS_DATA_PATH, "Adopters"),
        default_value=pd.DataFrame(),
        operation_name="adopters data loading"
    )
    
    return resources

# Load all resources
app_resources = initialize_app_resources()

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'adopter_profile': {},
        'profile_uuid': str(uuid.uuid4()),
        'recommendations_df': pd.DataFrame(),
        'all_scores_df': pd.DataFrame(),
        'find_matches_clicked': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# --- UI Helper Functions ---
def show_app_status():
    """Display app status and any critical errors."""
    if app_resources['errors']:
        st.error("⚠️ Some critical components failed to load:")
        for error in app_resources['errors']:
            st.error(f"• {error}")
        # Do not return False here, let it proceed to show OpenAI specific warnings if any
        # return False 
    
    # Check OpenAI client status specifically (already warned by get_openai_client if key missing)
    if app_resources['openai_client'] is None:
        # The get_openai_client function already shows a st.warning if key is missing.
        # We could add another message here if needed, or rely on its warning.
        logger.info("OpenAI client not initialized, explanations will be off.")
    
    if app_resources['adopters_df'].empty:
        st.warning("📊 Adopter data not loaded - using default options")
    
    return not app_resources['errors'] # Return True if no *critical* errors

def get_ui_options():
    """Get options for UI dropdowns from data or defaults."""
    adopters_df = app_resources['adopters_df']
    
    housing_options = (
        adopters_df['housing'].unique().tolist() 
        if 'housing' in adopters_df.columns and not adopters_df['housing'].empty
        else ['House', 'Apartment', 'Condo', 'Townhouse', 'Other']
    )
    
    activity_options = (
        adopters_df['activity'].unique().tolist()
        if 'activity' in adopters_df.columns and not adopters_df['activity'].empty  
        else ['Low', 'Moderate', 'High', 'Active', 'Very Active']
    )
    
    return housing_options, activity_options

def create_adopter_profile_ui():
    """Create the adopter profile input UI."""
    st.sidebar.header("🏠 Your Profile")
    
    housing_options, activity_options = get_ui_options()
    
    # Get current values from session state
    current_profile = st.session_state.adopter_profile
    
    # Basic info
    age = st.sidebar.number_input(
        "Your Age:", 
        min_value=18, max_value=100, 
        value=current_profile.get('age', 30),
        help="Your age helps us match you with appropriate pets"
    )
    
    household_size = st.sidebar.number_input(
        "Household Size:", 
        min_value=1, max_value=10,
        value=current_profile.get('household_size', 2),
        help="Number of people in your household"
    )
    
    # Housing
    housing_current = current_profile.get('housing', housing_options[0])
    housing_idx = housing_options.index(housing_current) if housing_current in housing_options else 0
    housing = st.sidebar.selectbox(
        "Housing Type:", 
        options=housing_options, 
        index=housing_idx,
        help="Your living situation affects pet suitability"
    )
    
    # Activity level
    activity_current = current_profile.get('activity', activity_options[0])
    activity_idx = activity_options.index(activity_current) if activity_current in activity_options else 0
    activity = st.sidebar.selectbox(
        "Activity Level:", 
        options=activity_options,
        index=activity_idx,
        help="Your activity level helps match pet energy needs"
    )
    
    # Pet experience
    has_prior_pets = st.sidebar.radio(
        "Have you owned pets before?",
        options=['Yes', 'No'],
        index=0 if current_profile.get('has_prior_pets', False) else 1,
        help="Experience with pets affects recommendations"
    )
    
    return {
        'age': age,
        'household_size': household_size,
        'housing': housing,
        'activity': activity,
        'has_prior_pets': has_prior_pets == 'Yes'
    }

def display_pet_recommendation(pet_data: pd.Series, index: int):
    """Display a single pet recommendation."""
    raw_pet_name = pet_data.get('Name')
    # animal_id = pet_data.get('Animal ID', 'Unknown ID') # Animal ID no longer needed for default name

    invalid_names_to_check = ["nan", "unknown", "no name", "", None]

    if pd.isna(raw_pet_name) or (isinstance(raw_pet_name, str) and raw_pet_name.strip().lower() in invalid_names_to_check):
        pet_name_cleaned = "Unnamed Pet" # Hardcoded default name
    else:
        pet_name_cleaned = str(raw_pet_name).encode('ascii', 'ignore').decode('ascii').strip()
        if not pet_name_cleaned: # If cleaning results in an empty string, also use default
            pet_name_cleaned = "Unnamed Pet"
    
    st.markdown(f"### 🐾 {pet_name_cleaned}")
    
    # Pet details in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Basic Info:**
        - **Type:** {pet_data.get('Animal Type', 'N/A')}
        - **Breed:** {pet_data.get('Breed', 'N/A')}
        - **Age:** {pet_data.get('Age upon Outcome', 'N/A')}
        - **Sex:** {pet_data.get('Sex upon Outcome', 'N/A')}
        """)
    
    with col2:
        st.markdown(f"""
        **Details:**
        - **Size:** {pet_data.get('Size', 'N/A')}
        - **Color:** {pet_data.get('Color', 'N/A')}
        - **ID:** {pet_data.get('Animal ID', 'N/A')}
        """)
    
    # Match score
    score = pet_data.get('score', 0.0)
    st.metric(
        label="🎯 Compatibility Score", 
        value=f"{score:.1%}",
        help="Higher scores indicate better compatibility"
    )
    
    # LLM explanation (now OpenAI)
    if app_resources['openai_client']:
        with st.spinner("Generating explanation..."):
            explanation = generate_match_explanation(
                st.session_state.adopter_profile,
                pet_data,
                app_resources['openai_client'],
                str(pet_data.get('Animal ID', f'pet_{index}'))
            )
            st.info(f"**Why this is a great match:** {explanation}")

# --- Main App UI ---
def main():
    """Main application UI."""
    st.title("🐾 AI Pet Match Recommender")
    st.markdown("*Find your perfect companion with AI-powered recommendations*")
    
    # Check app status
    if not show_app_status():
        st.stop()
    
    # Show data stats
    pets_count = len(app_resources['pets_df'])
    st.sidebar.markdown(f"📊 **{pets_count:,} pets** available for matching")
    
    # Create profile UI
    profile_data = create_adopter_profile_ui()
    
    # Find matches button
    if st.sidebar.button("✨ Find My Perfect Match!", type="primary", use_container_width=True):
        # Update session state
        st.session_state.adopter_profile = {
            **profile_data,
            'profile_uuid': st.session_state.profile_uuid
        }
        st.session_state.find_matches_clicked = True
        
        # Perform matching
        with st.spinner("🔍 Searching through thousands of pets..."):
            try:
                recommendations, all_scores = rank_pets_batch(
                    st.session_state.adopter_profile,
                    app_resources['pets_df'],
                    app_resources['rec_model'],
                    Config.TOP_K_RECOMMENDATIONS
                )
                
                st.session_state.recommendations_df = recommendations
                st.session_state.all_scores_df = all_scores
                
                if not recommendations.empty:
                    st.success(f"🎉 Found {len(recommendations)} great matches for you!")
                else:
                    st.warning("😕 No matches found. Try adjusting your preferences.")
                    
            except Exception as e:
                logger.error(f"Error during matching: {e}", exc_info=True)
                st.error("Sorry, something went wrong during matching. Please try again.")
    
    # Display recommendations
    if not st.session_state.recommendations_df.empty:
        st.markdown("---")
        st.subheader("🌟 Your Top Pet Matches")
        
        recommendations = st.session_state.recommendations_df
        
        # Create columns for layout
        if len(recommendations) == 1:
            display_pet_recommendation(recommendations.iloc[0], 0)
        else:
            cols = st.columns(min(len(recommendations), 3))
            for i, (_, pet_data) in enumerate(recommendations.iterrows()):
                with cols[i % len(cols)]:
                    display_pet_recommendation(pet_data, i)
                    if i < len(recommendations) - 1:
                        st.markdown("---")
    
    elif st.session_state.find_matches_clicked:
        st.info("👆 Click the button above to find your perfect pet match!")
    
    # Debug info (only if enabled)
    if os.getenv("STREAMLIT_DEBUG_MODE") == "true" and not st.session_state.all_scores_df.empty:
        with st.expander("🔧 Debug Information: Score Analysis", expanded=False):
            all_scores_df = st.session_state.all_scores_df
            st.subheader("Score Distribution Statistics")
            st.dataframe(all_scores_df['score'].describe().to_frame().T) # Show full describe() output

            st.subheader("Score Histogram")
            # Using Altair for a simple histogram, as st.pyplot can be heavy
            try:
                import altair as alt
                chart = alt.Chart(all_scores_df).mark_bar().encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=50), title="Match Score"),
                    alt.Y('count()', title="Number of Pets")
                ).properties(
                    title='Distribution of Match Scores in Sampled Batch'
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.caption("Altair library not found. Skipping histogram. Add 'altair' to requirements to see this.")
            except Exception as e:
                st.caption(f"Could not generate histogram: {e}")

            st.subheader(f"Top Scoring Pets in Sample (Max {Config.BATCH_SIZE})")
            # Display top N, e.g., top 10 or all if less than 10
            num_to_show = min(10, len(all_scores_df))
            st.dataframe(all_scores_df.head(num_to_show)[
                ['Animal ID', 'Name', 'Animal Type', 'Breed', 'Age upon Outcome', 'Size', 'score']
            ])
            
            # Show details of any perfect/near-perfect scores found
            perfect_threshold_debug = 0.999
            perfect_scores_debug_df = all_scores_df[all_scores_df['score'] >= perfect_threshold_debug]
            if not perfect_scores_debug_df.empty:
                st.subheader(f"Pets with Score >= {perfect_threshold_debug}")
                st.dataframe(perfect_scores_debug_df[
                    ['Animal ID', 'Name', 'Animal Type', 'Breed', 'Age upon Outcome', 'Size', 'score']
                ])
            
            st.caption(f"Displaying analysis for a sample of {len(all_scores_df)} pets.")

if __name__ == "__main__":
    main()
