#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for parsing and default values
DEFAULT_PET_AGE_YEARS = 1.0
DEFAULT_PET_ACTIVITY = 'moderate'
DEFAULT_PET_SIZE = 'medium'
DEFAULT_NEEDS_EXPERIENCED_OWNER = False
DEFAULT_GOOD_WITH_CHILDREN = True

# Constants for scoring logic
AGE_PET_TO_HUMAN_YEARS_SCALE = 30.0
AGE_DIFF_PENALTY_FACTOR = 0.1
ACTIVITY_DIFF_PENALTY_FACTOR = 1.0 # Applied as (1.0 + diff)
MATCH_SCORE_THRESHOLD = 0.7

# Housing scores (example, can be externalized if complex)
HOUSING_SCORES = {
    ('apartment', 'small'): 1.0,
    ('apartment', 'medium'): 0.7,
    ('apartment', 'large'): 0.3,
    ('house', 'small'): 1.0,
    ('house', 'medium'): 1.0,
    ('house', 'large'): 0.9,
    ('condo', 'small'): 1.0, # Assuming condo is similar to apartment/house
    ('condo', 'medium'): 0.8,
    ('condo', 'large'): 0.5,
}
DEFAULT_HOUSING_SCORE = 0.5

# Experience scores
EXP_SCORE_NEEDS_EXP_NO_PRIOR = 0.3
EXP_SCORE_NEEDS_EXP_HAS_PRIOR = 1.0
EXP_SCORE_NO_NEEDS_HAS_PRIOR = 0.9
EXP_SCORE_NO_NEEDS_NO_PRIOR = 0.8 # was 0.8, could also be 0.7 if no prior experience means less ideal

# Household scores
HH_SCORE_HAS_CHILDREN_GOOD_WITH = 1.0
HH_SCORE_HAS_CHILDREN_NOT_GOOD = 0.2
HH_SCORE_NO_CHILDREN_GOOD_WITH = 0.9 # Pet good with children, but no children, still positive
HH_SCORE_NO_CHILDREN_NOT_GOOD = 0.8 # Pet not good with children, no children, neutral/slightly positive


def parse_age_years(age_str: str) -> float:
    """Parse age string (e.g., '2 years' or '6 months') into years."""
    if pd.isna(age_str):
        return DEFAULT_PET_AGE_YEARS
        
    match = re.match(r'(\\d+)\\s+(year|month)s?', str(age_str).lower())
    if not match:
        return DEFAULT_PET_AGE_YEARS
        
    number, unit = match.groups()
    if unit == 'year':
        return float(number)
    else:  # months
        return float(number) / 12.0

def calculate_age_score(adopter_age: float, pet_age_str: str) -> float:
    """Calculate compatibility score based on adopter and pet ages."""
    pet_age_years = parse_age_years(pet_age_str)
    # Rough scaling of pet age to human years for comparison
    # Giving more weight to younger pets being more adaptable, and older pets matching older adopters
    # This is a heuristic and can be refined.
    # A younger adopter might be fine with an older pet, but an older adopter might struggle with a very young, energetic pet.
    # Consider pet_age_years == 0 (e.g. <1 month old)
    if pet_age_years == 0: # Avoid division by zero or extreme scaling for very young pets
        effective_pet_age_comparison = 1 # treat as 1 human year for comparison
    else:
        effective_pet_age_comparison = adopter_age / (AGE_PET_TO_HUMAN_YEARS_SCALE * pet_age_years)
    
    # Score higher if adopter_age is a multiple of scaled pet age (e.g. adopter 30, pet 1 (scaled 30))
    # The original age_diff was abs(adopter_age - 30*pet_age_years)
    # Let's try a slightly different approach:
    # Ideal match if adopter_age is around 20-40 "human years" equivalent of pet.
    # This is highly heuristic. The original was simpler. Let's revert to a penalized difference.
    age_diff = abs(adopter_age - (AGE_PET_TO_HUMAN_YEARS_SCALE * pet_age_years) )
    return 1.0 / (1.0 + AGE_DIFF_PENALTY_FACTOR * age_diff)


def calculate_activity_score(adopter_activity: str, pet_activity_needs: str) -> float:
    """Calculate compatibility score based on activity levels."""
    activity_levels = {'low': 1, 'moderate': 2, 'high': 3, 'active': 3, 'very active':3} # added synonyms
    # Use .get with a default to handle potential None or unexpected values safely
    adopter_level = activity_levels.get(str(adopter_activity).lower(), 2) # Default to moderate
    pet_level = activity_levels.get(str(pet_activity_needs).lower(), 2) # Default to moderate
    
    diff = abs(adopter_level - pet_level)
    return 1.0 / (1.0 + ACTIVITY_DIFF_PENALTY_FACTOR * diff) # Using factor for consistency, though it's 1.0

def calculate_housing_score(adopter_housing: str, pet_size: str) -> float:
    """Calculate compatibility score based on housing type and pet size."""
    # Default to medium if size is unknown
    if pd.isna(pet_size) or str(pet_size).lower() not in ['small', 'medium', 'large', 'extra large', 'xl']: # added more size variants
        effective_pet_size = DEFAULT_PET_SIZE
    else:
        effective_pet_size = str(pet_size).lower()
        if effective_pet_size == 'extra large' or effective_pet_size == 'xl':
            effective_pet_size = 'large' # Map XL to large for scoring table
            
    return HOUSING_SCORES.get((str(adopter_housing).lower(), effective_pet_size), DEFAULT_HOUSING_SCORE)

def calculate_experience_score(has_prior_pet: bool, pet_needs_experienced_owner: bool) -> float:
    """Calculate compatibility score based on pet ownership experience."""
    if pet_needs_experienced_owner and not has_prior_pet:
        return EXP_SCORE_NEEDS_EXP_NO_PRIOR
    elif pet_needs_experienced_owner and has_prior_pet:
        return EXP_SCORE_NEEDS_EXP_HAS_PRIOR
    elif not pet_needs_experienced_owner and has_prior_pet: # Pet doesn't need experience, adopter has it
        return EXP_SCORE_NO_NEEDS_HAS_PRIOR
    else: # Pet doesn't need experience, adopter doesn't have it
        return EXP_SCORE_NO_NEEDS_NO_PRIOR

def calculate_household_score(household_size: int, good_with_children: bool, pet_age_str: str) -> float:
    """Calculate compatibility score based on household composition, considering pet age for 'children' interaction."""
    # Consider very young pets (puppies/kittens) generally good with children if socialized.
    pet_age_years = parse_age_years(pet_age_str)
    
    # Assuming household_size > 2 implies potential for children, or at least a busier household.
    # The README says "household_size > 2", but this could mean 2 adults + 1 child, or just 3 adults.
    # Let's refine: assume children if household_size implies more than 2 adults or explicitly stated.
    # For this rule, we'll stick to the simpler "household_size > 2" as a proxy for children for now.
    has_children_proxy = household_size > 2 
    
    # If pet is very young, it's more adaptable, slightly increasing score if good_with_children is False but children are present.
    age_leniency_factor = 0.0
    if pet_age_years < 0.5 and not good_with_children and has_children_proxy: # e.g. puppy/kitten
        age_leniency_factor = 0.2 # Small boost for adaptability of young pets

    if has_children_proxy and good_with_children:
        return HH_SCORE_HAS_CHILDREN_GOOD_WITH
    elif has_children_proxy and not good_with_children:
        return HH_SCORE_HAS_CHILDREN_NOT_GOOD + age_leniency_factor # Apply leniency
    elif not has_children_proxy and good_with_children: # No children, pet is good with them (neutral to good)
        return HH_SCORE_NO_CHILDREN_GOOD_WITH
    else: # No children, pet not necessarily good with them (neutral)
        return HH_SCORE_NO_CHILDREN_NOT_GOOD


def calculate_match_score(adopter_row: pd.Series, pet_row: pd.Series) -> float:
    """Calculate overall match score between an adopter and a pet."""
    scores = []
    weights = [] # Optional: for weighted average
    
    # Age compatibility
    # Adopter age is adopter_row['age'] or adopter_row['adopter_age']
    # Pet age is pet_row['Age upon Outcome'] or pet_row['pet_age_parsed_years']
    age_score = calculate_age_score(adopter_row['age'], pet_row['Age upon Outcome'])
    scores.append(age_score)
    weights.append(1.0) 
    
    # Activity level compatibility
    # Adopter activity: adopter_row['activity'] or adopter_row['adopter_activity_level']
    # Pet activity needs: pet_row.get('activity_needs', DEFAULT_PET_ACTIVITY) or 'pet_activity_needs'
    activity_score = calculate_activity_score(
        adopter_row['activity'], 
        pet_row.get('activity_needs', DEFAULT_PET_ACTIVITY) # pet_df might not have this column
    )
    scores.append(activity_score)
    weights.append(1.0)
    
    # Housing compatibility
    # Adopter housing: adopter_row['housing'] or adopter_row['adopter_housing_type']
    # Pet size: Adopter's preference is used: adopter_row['preferred_size']
    # This was noted in README: "adopter_row['preferred_size']" was used. This implies pets are filtered by size preference first OR pet_size is from pet_row for actual compatibility.
    # The original code used adopter_row['preferred_size'] for the pet_size argument. This seems like a mismatch.
    # It should be pet_row['Size'] or similar. The README states "Adaptation of the pairing script to handle data variability, such as the absence of a "Weight (lbs)" column in the pets_silver data. This involved removing weight-dependent features and adjusting the scoring logic to neutralize the impact of the missing size component."
    # Let's assume 'Size' column exists on pet_row, or a default if not.
    # If we use adopter's preferred_size, it's not pet compatibility but preference matching.
    # For direct compatibility, we should use pet's actual size.
    # Let's use pet_row.get('Size', DEFAULT_PET_SIZE) for now and assume 'Size' is a column in pets_df
    housing_score = calculate_housing_score(
        adopter_row['housing'], 
        pet_row.get('Size', DEFAULT_PET_SIZE) # Use actual pet size
    )
    scores.append(housing_score)
    weights.append(1.0)
    
    # Experience compatibility
    # Adopter prior pet: adopter_row['has_prior_pet'] or adopter_row['adopter_has_owned_pets']
    # Pet needs experience: pet_row.get('needs_experienced_owner', DEFAULT_NEEDS_EXPERIENCED_OWNER) or 'pet_needs_experienced_owner'
    experience_score = calculate_experience_score(
        adopter_row['has_prior_pet'], 
        pet_row.get('needs_experienced_owner', DEFAULT_NEEDS_EXPERIENCED_OWNER)
    )
    scores.append(experience_score)
    weights.append(1.0)
    
    # Household compatibility
    # Adopter household size: adopter_row['household_size'] or adopter_row['adopter_household_size']
    # Pet good with children: pet_row.get('good_with_children', DEFAULT_GOOD_WITH_CHILDREN) or 'pet_good_with_children'
    household_score = calculate_household_score(
        adopter_row['household_size'],
        pet_row.get('good_with_children', DEFAULT_GOOD_WITH_CHILDREN),
        pet_row['Age upon Outcome'] # Pass pet age for context
    )
    scores.append(household_score)
    weights.append(1.0)
    
    # Calculate weighted average (equal weights for now)
    if not scores: # Should not happen if all scores are appended
        return 0.0
    return np.average(scores, weights=weights if sum(weights) > 0 else None)


def generate_training_pairs(adopters_df: pd.DataFrame, pets_df: pd.DataFrame, 
                          sample_size: int = 10000) -> pd.DataFrame:
    """Generate training pairs by sampling from all possible adopter-pet combinations."""
    logger.info("Generating training pairs...")
    
    # Standardize column names before sampling/pairing if they are different
    # For now, assume input DFs use names expected by calculate_match_score's row access.
    
    # Sample to keep manageable - ensure n is not greater than population
    n_adopters_to_sample = min(len(adopters_df), int(np.sqrt(sample_size)))
    n_pets_to_sample = min(len(pets_df), int(np.sqrt(sample_size)))

    if n_adopters_to_sample == 0 or n_pets_to_sample == 0:
        logger.warning("Not enough adopters or pets to sample. Returning empty DataFrame.")
        return pd.DataFrame()

    adopter_samples = adopters_df.sample(n=n_adopters_to_sample, random_state=42)
    pet_samples = pets_df.sample(n=n_pets_to_sample, random_state=42)
    
    pairs = []
    for _, adopter in adopter_samples.iterrows():
        for _, pet in pet_samples.iterrows():
            score = calculate_match_score(adopter, pet)
            
            # Standardized output column names
            pair = {
                'adopter_id': adopter['adopter_id'],
                'pet_id': pet['Animal ID'], # Assuming 'Animal ID' is the pet identifier
                'adopter_age': adopter['age'],
                'adopter_housing_type': adopter['housing'],
                'adopter_activity_level': adopter['activity'],
                'adopter_has_prior_pets': adopter['has_prior_pet'], # Changed from has_owned_pets
                'adopter_household_size': adopter['household_size'],
                'pet_age_raw': pet['Age upon Outcome'], # Keep raw age string
                'pet_age_years': parse_age_years(pet['Age upon Outcome']), # Add parsed age
                'pet_animal_type': pet['Animal Type'],
                'pet_breed': pet['Breed'],
                'pet_color': pet['Color'],
                'pet_size': pet.get('Size', DEFAULT_PET_SIZE), # Add pet's actual size used in scoring
                'pet_activity_needs': pet.get('activity_needs', DEFAULT_PET_ACTIVITY),
                'pet_needs_experienced_owner': pet.get('needs_experienced_owner', DEFAULT_NEEDS_EXPERIENCED_OWNER),
                'pet_good_with_children': pet.get('good_with_children', DEFAULT_GOOD_WITH_CHILDREN),
                'match_score': score,
                'match_label': 1 if score >= MATCH_SCORE_THRESHOLD else 0
            }
            pairs.append(pair)
    
    if not pairs:
        logger.warning("No pairs were generated. Returning empty DataFrame.")
        return pd.DataFrame()
        
    pairs_df = pd.DataFrame(pairs)
    
    # Balance the dataset
    pos_samples = pairs_df[pairs_df['match_label'] == 1]
    neg_samples_pool = pairs_df[pairs_df['match_label'] == 0]
    
    if len(pos_samples) == 0:
        logger.warning("No positive samples generated. Cannot balance. Returning all generated pairs or empty if none.")
        return pairs_df # Or handle as an error / return empty based on requirements
    
    num_neg_to_sample = min(len(pos_samples), len(neg_samples_pool))
    
    if num_neg_to_sample > 0 :
        neg_samples = neg_samples_pool.sample(n=num_neg_to_sample, random_state=42)
        balanced_pairs = pd.concat([pos_samples, neg_samples], ignore_index=True)
    elif len(pos_samples) > 0 : # Only positive samples, no negative ones to sample
        logger.warning("Only positive samples found. Dataset will not be balanced.")
        balanced_pairs = pos_samples
    else: # Should be caught by earlier checks
        logger.warning("No positive or negative samples to form a balanced dataset.")
        return pd.DataFrame()

    logger.info(f"Generated {len(balanced_pairs)} balanced training pairs (Positive: {len(pos_samples)}, Negative: {num_neg_to_sample if num_neg_to_sample > 0 else 0})")
    return balanced_pairs.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle


def main(bucket: str):
    """Main function to generate and save training pairs."""
    logger.info(f"Starting pair generation process with bucket: {bucket}")
    
    # Construct GCS paths safely
    # Assuming bucket format gs://bucket-name, no trailing slash needed for read_parquet if it expects a directory
    adopters_gcs_path = os.path.join(bucket, "Adopters") # pd.read_parquet can read a directory of parquet files
    pets_gcs_path = os.path.join(bucket, "pets_silver")
    output_gcs_path = os.path.join(bucket, "train_pairs.parquet")

    logger.info(f"Reading adopters data from GCS path: {adopters_gcs_path}")
    try:
        adopters_df = pd.read_parquet(adopters_gcs_path)
    except Exception as e:
        logger.error(f"Failed to read adopters data from {adopters_gcs_path}: {e}")
        return
    logger.info(f"Adopters columns: {adopters_df.columns.tolist()}")
    logger.info(f"Adopters data sample:\\n{adopters_df.head()}")

    logger.info(f"Reading pets data from GCS path: {pets_gcs_path}")
    try:
        pets_df = pd.read_parquet(pets_gcs_path)
    except Exception as e:
        logger.error(f"Failed to read pets data from {pets_gcs_path}: {e}")
        return
    logger.info(f"Pets columns: {pets_df.columns.tolist()}")
    logger.info(f"Pets data sample:\\n{pets_df.head()}")

    # Data validation and cleaning (minimal example)
    required_adopter_cols = ['adopter_id', 'age', 'housing', 'activity', 'has_prior_pet', 'household_size']
    required_pet_cols = ['Animal ID', 'Age upon Outcome', 'Animal Type', 'Breed', 'Color'] # 'Size' is handled by .get()

    for col in required_adopter_cols:
        if col not in adopters_df.columns:
            logger.error(f"Missing required column '{col}' in adopters data. Aborting.")
            return
    for col in required_pet_cols:
        if col not in pets_df.columns:
            logger.error(f"Missing required column '{col}' in pets data. Aborting.")
            return
            
    # Ensure key columns for scoring exist or have defaults handled by .get() in scoring functions
    # Example: if 'activity_needs' is crucial for pets_df, check or ensure robust handling.
    # The .get() provides robustness.

    # Generate training pairs
    train_pairs = generate_training_pairs(adopters_df, pets_df)
    
    if train_pairs.empty:
        logger.warning("No training pairs were generated. Output file will not be created.")
        return

    # Write output to GCS
    logger.info(f"Writing {len(train_pairs)} training pairs to {output_gcs_path}")
    try:
        train_pairs.to_parquet(output_gcs_path, index=False)
    except Exception as e:
        logger.error(f"Failed to write training pairs to {output_gcs_path}: {e}")
        return
    
    logger.info("✓ Pair generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training pairs for pet-adopter matching")
    parser.add_argument("--bucket", required=True, help="GCS bucket path (e.g., gs://bucket-name)")
    args = parser.parse_args()
    main(args.bucket)