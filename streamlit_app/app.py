#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Pet Adoption Matcher - Simplified")

# --- Application Constants & Configuration ---
PETS_DATA_PATH = "data/pets_silver_local/pets_silver"
ADOPTERS_DATA_PATH = "data/adopters_local/Adopters"
TOP_K_RECOMMENDATIONS = 5

@st.cache_data
def load_parquet_data(data_path: str, data_name: str) -> pd.DataFrame:
    logger.info(f"Loading {data_name} data from {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"{data_name} data path does not exist: {data_path}")
        st.error(f"{data_name} data path does not exist: {data_path}. Please check the path.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"{data_name} data loaded successfully from {data_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {data_name} data from {data_path}: {e}")
        st.error(f"Error loading {data_name} data from {data_path}: {e}")
        return pd.DataFrame()

# Simple rule-based matching function (replaces ML model)
def simple_pet_matching(adopter_profile: dict, pets_df: pd.DataFrame, top_k: int = TOP_K_RECOMMENDATIONS):
    """Simple rule-based pet matching without ML model."""
    if pets_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original
    matched_pets = pets_df.copy()
    matched_pets['match_score'] = 0.0
    
    # Simple scoring based on adopter preferences
    for idx, pet in matched_pets.iterrows():
        score = 0.0
        
        # Animal type preference
        pet_type = str(pet.get('Animal Type', '')).lower()
        if adopter_profile.get('preferred_animal') == 'Any' or adopter_profile.get('preferred_animal', '').lower() in pet_type:
            score += 3
            
        # Activity level matching (simplified)
        adopter_activity = adopter_profile.get('activity', 'Moderate')
        if adopter_activity == 'High':
            score += 2  # Assume all pets benefit from active owners
        elif adopter_activity == 'Moderate':
            score += 1
            
        # Experience with pets
        if adopter_profile.get('has_prior_pets') == True:
            score += 1
            
        # Random component for variety
        score += np.random.random() * 2
        
        matched_pets.at[idx, 'match_score'] = score
    
    # Sort by match score and return top K
    top_matches = matched_pets.nlargest(top_k, 'match_score')
    return top_matches

def main():
    st.title("🐕 Pet Adoption Matcher - Demo Version")
    st.markdown("*Note: This is a simplified version without ML recommendations for deployment testing.*")
    
    # Load data
    pets_df = load_parquet_data(PETS_DATA_PATH, "Pets")
    adopters_df = load_parquet_data(ADOPTERS_DATA_PATH, "Adopters")
    
    if pets_df.empty:
        st.error("No pet data available. Please check data files.")
        return
        
    st.sidebar.header("🏠 Your Preferences")
    
    # Adopter profile input
    adopter_profile = {}
    
    with st.sidebar:
        adopter_profile['age'] = st.slider("Your Age", 18, 80, 30)
        adopter_profile['household_size'] = st.selectbox("Household Size", [1, 2, 3, 4, 5, 6])
        adopter_profile['housing'] = st.selectbox("Housing Type", 
                                                 ["Apartment", "House", "Condo", "Other"])
        adopter_profile['activity'] = st.selectbox("Activity Level", 
                                                  ["Low", "Moderate", "High"])
        adopter_profile['has_prior_pets'] = st.checkbox("I have experience with pets")
        adopter_profile['preferred_animal'] = st.selectbox("Preferred Animal Type",
                                                          ["Any", "Dog", "Cat", "Bird", "Other"])
    
    st.header("🎯 Your Top Pet Matches")
    
    if st.button("Find My Matches", type="primary"):
        with st.spinner("Finding your perfect matches..."):
            matches = simple_pet_matching(adopter_profile, pets_df, TOP_K_RECOMMENDATIONS)
            
            if matches.empty:
                st.warning("No matches found. Try adjusting your preferences.")
                return
                
            # Display matches
            for idx, (_, pet) in enumerate(matches.iterrows()):
                with st.expander(f"🐾 Match #{idx + 1}: {pet.get('Name', 'Unnamed Pet')}", expanded=(idx == 0)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        **Animal Type:** {pet.get('Animal Type', 'Unknown')}  
                        **Breed:** {pet.get('Breed', 'Mixed')}  
                        **Age:** {pet.get('Age upon Outcome', 'Unknown')}  
                        **Color:** {pet.get('Color', 'Unknown')}  
                        **Size:** {pet.get('Size', 'Unknown')}
                        """)
                        
                    with col2:
                        match_score = pet.get('match_score', 0)
                        st.metric("Match Score", f"{match_score:.1f}/10")
                        
                        st.markdown(f"""
                        **Why this might be a good match:**
                        - This {pet.get('Animal Type', 'pet').lower()} could be a great companion
                        - Your {adopter_profile['activity'].lower()} activity level is noted
                        - {"Perfect for experienced owners!" if adopter_profile['has_prior_pets'] else "Great for new pet parents!"}
                        """)
    
    # Show data statistics
    st.header("📊 Available Pets Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pets", len(pets_df))
    
    with col2:
        if 'Animal Type' in pets_df.columns:
            dogs = len(pets_df[pets_df['Animal Type'].str.contains('Dog', na=False)])
            st.metric("Dogs", dogs)
    
    with col3:
        if 'Animal Type' in pets_df.columns:
            cats = len(pets_df[pets_df['Animal Type'].str.contains('Cat', na=False)])
            st.metric("Cats", cats)
    
    # Show sample pets
    if st.checkbox("Show all available pets"):
        st.dataframe(pets_df.head(20))

if __name__ == "__main__":
    main() 