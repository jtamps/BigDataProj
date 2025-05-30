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
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", os.path.join(SCRIPT_DIR, "models/llama-3-8b-instruct.Q4_K_M.gguf"))
    
    # Model parameters
    TOP_K_RECOMMENDATIONS = 3
    BATCH_SIZE = 2000
    DEFAULT_PET_AGE_YEARS = 1.0
    
    # Llama parameters
    N_GPU_LAYERS = -1
    N_CTX = 2048
    
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
        raise

def safe_operation(func, default_value=None, operation_name: str = "operation"):
    """Safely execute an operation with error handling."""
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        return default_value

# --- Llama Availability Check ---
@st.cache_resource
def check_llama_availability() -> Tuple[bool, Optional[type]]:
    """Check if llama_cpp is available, with caching to avoid repeated checks."""
    try:
        from llama_cpp import Llama
        logger.info("llama_cpp is available. LLM explanations will be enabled if model is found.")
        return True, Llama
    except ImportError:
        logger.warning("llama_cpp not available. LLM explanations will be disabled.")
        return False, None

LLAMA_AVAILABLE, Llama = check_llama_availability()

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

# --- Model and Data Loading ---
@st.cache_resource
def load_llama_model(model_path: str) -> Optional[object]:
    """Load Llama model with comprehensive error handling."""
    if not LLAMA_AVAILABLE:
        logger.info("Llama not available, skipping model load.")
        return None
    
    if not os.path.exists(model_path):
        logger.warning(f"Llama model not found at {model_path}")
        return None
    
    with error_handler("Llama model loading", show_user_error=False):
        logger.info(f"Loading Llama model from {model_path}")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=Config.N_GPU_LAYERS,
            n_ctx=Config.N_CTX,
            verbose=False
        )
        logger.info("Llama model loaded successfully.")
        return llm
    
    return None

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
    return top_recommendations, ranked_pets

# --- LLM Explanation Generation ---
@st.cache_data
def generate_match_explanation(_adopter_profile: Dict, _pet_details: pd.Series, 
                             _llm_model: object, pet_id: str) -> str:
    """Generate LLM explanation with error handling."""
    if not _llm_model:
        return "LLM explanations are currently unavailable."
    
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
        
        # Create prompt
        prompt = f"""A {adopter_info['age']}-year-old person with a {adopter_info['household_size']}-person household, living in a {adopter_info['housing']}, with {adopter_info['activity']} activity level, who has {adopter_info['has_prior_pets']} owned pets before, is being matched with {pet_info['name']}, a {pet_info['age']} {pet_info['type']} ({pet_info['breed']}, {pet_info['size']}).

Briefly explain in 2-3 sentences why this could be a great match, focusing on compatibility."""
        
        # Generate explanation
        completion = _llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful pet adoption counselor. Be positive and concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        explanation = completion['choices'][0]['message']['content'].strip()
        logger.info(f"Generated explanation for pet {pet_id}")
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation for pet {pet_id}: {e}")
        return "Could not generate explanation at this time."

# --- Initialize App Resources ---
@st.cache_resource
def initialize_app_resources():
    """Initialize all app resources with error handling."""
    resources = {
        'llm_model': None,
        'rec_model': None,
        'pets_df': pd.DataFrame(),
        'adopters_df': pd.DataFrame(),
        'errors': []
    }
    
    # Load LLM model
    resources['llm_model'] = safe_operation(
        lambda: load_llama_model(Config.LLAMA_MODEL_PATH),
        default_value=None,
        operation_name="LLM model loading"
    )
    
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
        st.error("⚠️ Some components failed to load:")
        for error in app_resources['errors']:
            st.error(f"• {error}")
        return False
    
    # Show warnings for optional components
    if app_resources['llm_model'] is None:
        st.warning("💬 LLM explanations are not available (this is optional)")
    
    if app_resources['adopters_df'].empty:
        st.warning("📊 Adopter data not loaded - using default options")
    
    return True

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
    pet_name = pet_data.get('Name', f"Pet #{pet_data.get('Animal ID', 'Unknown')}")
    
    st.markdown(f"### 🐾 {pet_name}")
    
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
    
    # LLM explanation
    if app_resources['llm_model']:
        with st.spinner("Generating explanation..."):
            explanation = generate_match_explanation(
                st.session_state.adopter_profile,
                pet_data,
                app_resources['llm_model'],
                str(pet_data.get('Animal ID', f'pet_{index}'))
            )
            st.info(f"**Why this is a great match:** {explanation}")
    else:
        st.info("💬 _Detailed explanations would appear here if LLM was available_")

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
        with st.expander("🔧 Debug Information"):
            st.subheader("All Pet Scores (Sample)")
            st.dataframe(st.session_state.all_scores_df.head(20))
            st.caption(f"Showing sample of {len(st.session_state.all_scores_df)} scored pets")

if __name__ == "__main__":
    main()
