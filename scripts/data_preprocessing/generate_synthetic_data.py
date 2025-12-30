"""
Synthetic Data Generation for Testing and Development.
This script generates various types of synthetic data to support model development and testing.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any, Tuple
import argparse
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("synthetic_data_generator")

class SyntheticDataGenerator:
    """Generator for synthetic datasets."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_customer_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic customer data.
        
        Args:
            n_samples: Number of samples to generate.
        
        Returns:
            DataFrame containing customer data.
        """
        logger.info(f"Generating customer data for {n_samples} samples...")
        
        # Generate IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(n_samples)]
        
        # Generate Names
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_samples)]
        
        # Generate Age
        ages = np.clip(np.random.normal(35, 12, n_samples).astype(int), 18, 85)
        
        # Generate Gender
        genders = np.random.choice(["Male", "Female"], n_samples, p=[0.49, 0.51])
        
        # Generate Income
        income_levels = ["Low", "Medium", "High"]
        income = np.random.choice(income_levels, n_samples, p=[0.3, 0.5, 0.2])
        annual_income = np.where(
            income == "Low", np.random.uniform(25000, 50000, n_samples),
            np.where(
                income == "Medium", np.random.uniform(50000, 120000, n_samples),
                np.random.uniform(120000, 350000, n_samples)
            )
        ).astype(int)
        
        # Generate Location (US Cities for English context)
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
        cities_col = np.random.choice(cities, n_samples)
        regions = np.where(np.isin(cities_col, ["New York", "Philadelphia"]), "East", 
                  np.where(np.isin(cities_col, ["Los Angeles", "San Diego", "Phoenix"]), "West", "Central"))
        
        # Generate Customer Status
        customer_status = np.random.choice(["Active", "Inactive", "Churned"], n_samples, p=[0.7, 0.2, 0.1])
        
        # Generate Registration Date
        base_date = datetime(2020, 1, 1)
        registration_dates = [
            base_date + timedelta(days=int(np.random.exponential(365)))
            for _ in range(n_samples)
        ]
        
        # Generate Loyalty Score
        loyalty_scores = np.clip(np.random.normal(0.65, 0.2, n_samples), 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'name': names,
            'age': ages,
            'gender': genders,
            'income_level': income,
            'annual_income': annual_income,
            'city': cities_col,
            'region': regions,
            'customer_status': customer_status,
            'registration_date': registration_dates,
            'loyalty_score': np.round(loyalty_scores, 2)
        })
        
        logger.info("Customer data generated successfully")
        return df
    
    def generate_transaction_data(self, customer_ids: List[str], n_transactions: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic transaction data.
        
        Args:
            customer_ids: List of available customer IDs.
            n_transactions: Number of transactions to generate.
        
        Returns:
            DataFrame containing transaction data.
        """
        logger.info(f"Generating transaction data for {n_transactions} transactions...")
        
        # Generate Transaction IDs
        transaction_ids = [f"TRX_{i:08d}" for i in range(n_transactions)]
        
        # Randomly select customers
        selected_customers = np.random.choice(customer_ids, n_transactions)
        
        # Generate Dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 1, 1)
        date_range = (end_date - start_date).days
        transaction_dates = [
            start_date + timedelta(days=int(np.random.uniform(0, date_range)))
            for _ in range(n_transactions)
        ]
        
        # Generate Types
        transaction_types = ["Purchase", "Refund", "Subscription", "Payment", "Withdrawal"]
        types = np.random.choice(transaction_types, n_transactions, p=[0.65, 0.05, 0.2, 0.05, 0.05])
        
        # Generate Amounts based on type
        amounts = np.zeros(n_transactions)
        for i, t_type in enumerate(types):
            if t_type == "Purchase":
                amounts[i] = np.random.uniform(10, 1000)
            elif t_type == "Refund":
                amounts[i] = -np.random.uniform(10, 500)
            elif t_type == "Subscription":
                amounts[i] = np.random.uniform(10, 100)
            elif t_type == "Payment":
                amounts[i] = np.random.uniform(50, 2000)
            elif t_type == "Withdrawal":
                amounts[i] = -np.random.uniform(20, 1000)
        
        # Generate Categories
        categories = ["Electronics", "Clothing", "Grocery", "Entertainment", "Home & Garden", "Travel"]
        product_categories = np.random.choice(categories, n_transactions)
        
        # Generate Status
        status = np.random.choice(["Completed", "Pending", "Failed"], n_transactions, p=[0.94, 0.04, 0.02])
        
        # Generate Loyalty Points earned
        loyalty_points = np.where(
            status == "Completed",
            np.maximum(0, (amounts * 0.05).astype(int)),
            0
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'customer_id': selected_customers,
            'transaction_date': transaction_dates,
            'transaction_type': types,
            'amount': np.round(amounts, 2),
            'product_category': product_categories,
            'status': status,
            'loyalty_points': loyalty_points
        })
        
        # Sort by date
        df = df.sort_values('transaction_date').reset_index(drop=True)
        
        logger.info("Transaction data generated successfully")
        return df
    
    def generate_product_data(self, n_products: int = 500) -> pd.DataFrame:
        """
        Generate synthetic product data.
        
        Args:
            n_products: Number of products to generate.
        
        Returns:
            DataFrame containing product data.
        """
        logger.info(f"Generating product data for {n_products} products...")
        
        # Generate Product IDs
        product_ids = [f"PROD_{i:05d}" for i in range(n_products)]
        
        # Generate Product Names
        adjectives = ["Premium", "Luxury", "Essential", "Smart", "Wireless", "Portable", "Durable", "Eco-friendly"]
        nouns = ["Headphones", "Watch", "Smartphone", "Laptop", "Sneakers", "Backpack", "Camera", "Speaker", "Monitor", "Chair"]
        product_names = [f"{random.choice(adjectives)} {random.choice(nouns)}" for _ in range(n_products)]
        
        # Generate Categories
        categories = ["Electronics", "Clothing", "Grocery", "Entertainment", "Home", "Travel"]
        product_categories = np.random.choice(categories, n_products, p=[0.35, 0.2, 0.1, 0.1, 0.15, 0.1])
        
        # Generate Prices
        base_prices = {
            "Electronics": (100, 2500),
            "Clothing": (20, 300),
            "Grocery": (2, 50),
            "Entertainment": (10, 150),
            "Home": (30, 800),
            "Travel": (50, 1000)
        }
        
        prices = []
        for cat in product_categories:
            min_price, max_price = base_prices[cat]
            price = np.random.uniform(min_price, max_price)
            prices.append(round(price, 2))
        
        # Generate Costs (Margin simulation)
        costs = [price * np.random.uniform(0.4, 0.8) for price in prices]
        
        # Generate Stock Quantities
        quantities = np.random.poisson(100, n_products)
        quantities = np.maximum(quantities, 0) 
        
        # Generate Ratings
        ratings = np.clip(np.random.normal(4.2, 0.6, n_products), 1.0, 5.0)
        ratings = np.round(ratings, 1)
        
        # Generate Review Counts
        review_counts = np.random.poisson(80, n_products)
        review_counts = np.maximum(review_counts, 0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'product_id': product_ids,
            'product_name': product_names,
            'category': product_categories,
            'price': prices,
            'cost': np.round(costs, 2),
            'quantity_in_stock': quantities,
            'rating': ratings,
            'review_count': review_counts
        })
        
        logger.info("Product data generated successfully")
        return df
    
    def generate_medical_records(self, n_records: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic medical records.
        
        Args:
            n_records: Number of records to generate.
        
        Returns:
            DataFrame containing medical records.
        """
        logger.info(f"Generating medical records for {n_records} patients...")
        
        # Generate Patient IDs
        patient_ids = [f"PAT_{i:06d}" for i in range(n_records)]
        
        # Generate Names
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_records)]
        
        # Generate Age
        ages = np.clip(np.random.normal(48, 18, n_records).astype(int), 18, 95)
        
        # Generate Gender
        genders = np.random.choice(["Male", "Female"], n_records, p=[0.48, 0.52])
        
        # Generate Conditions
        medical_conditions_pool = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "Coronary Heart Disease", 
            "Osteoarthritis", "Migraine", "Major Depression", "Generalized Anxiety",
            "Obesity", "Hyperlipidemia", "GERD", "Hypothyroidism"
        ]
        conditions = []
        for _ in range(n_records):
            num_conditions = np.random.poisson(1.5)
            num_conditions = max(0, min(num_conditions, 5))
            patient_conditions = random.sample(medical_conditions_pool, num_conditions)
            conditions.append(", ".join(patient_conditions))
        
        # Generate Medications
        medications_pool = [
            "Lisinopril", "Metformin", "Albuterol", "Aspirin",
            "Ibuprofen", "Sertraline", "Atorvastatin", "Levothyroxine",
            "Omeprazole", "Amlodipine", "Losartan", "Gabapentin"
        ]
        medications_list = []
        for _ in range(n_records):
            num_medications = np.random.poisson(1.8)
            num_medications = max(0, min(num_medications, 6))
            patient_meds = random.sample(medications_pool, num_medications)
            medications_list.append(", ".join(patient_meds))
        
        # Generate Vitals
        systolic_bp = np.clip(np.random.normal(120 + 0.5 * ages, 15), 90, 210).astype(int)
        diastolic_bp = np.clip(np.random.normal(70 + 0.3 * ages, 10), 50, 130).astype(int)
        blood_sugar = np.clip(np.random.normal(95 + 1.0 * (ages > 50).astype(int) * 25, 20), 65, 350)
        
        # Generate Visit Dates
        visit_dates = [
            datetime(2023, 1, 1) + timedelta(days=int(np.random.uniform(0, 730)))
            for _ in range(n_records)
        ]
        
        # Generate Risk Score
        risk_scores = np.clip(
            0.2 * (ages > 60).astype(int) +
            0.5 * (np.array([len(c.split(',')) if c else 0 for c in conditions]) > 2).astype(int) +
            0.3 * (blood_sugar > 140).astype(int),
            0, 1
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'name': names,
            'age': ages,
            'gender': genders,
            'medical_conditions': conditions,
            'medications': medications_list,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'blood_sugar': np.round(blood_sugar, 1),
            'visit_date': visit_dates,
            'risk_score': np.round(risk_scores, 2)
        })
        
        logger.info("Medical records generated successfully")
        return df
    
    def generate_legal_documents(self, n_documents: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic legal documents.
        
        Args:
            n_documents: Number of documents to generate.
        
        Returns:
            List of dictionaries containing legal documents.
        """
        logger.info(f"Generating legal documents for {n_documents} documents...")
        
        document_types = ["Service Agreement", "Employment Contract", "Non-Disclosure Agreement (NDA)", "Last Will and Testament", "Lease Agreement", "Merger Plan"]
        parties = ["TechCorp Inc.", "Global Solutions Ltd.", "John Doe", "Jane Smith", "State Department", "Innovate LLC"]
        jurisdictions = ["California", "New York", "Texas", "Delaware", "United Kingdom", "European Union"]
        
        documents = []
        for i in range(n_documents):
            doc_type = random.choice(document_types)
            title = f"{doc_type} between {random.choice(parties)} and {random.choice(parties)}"
            jurisdiction = random.choice(jurisdictions)
            date = datetime(2021, 1, 1) + timedelta(days=int(np.random.uniform(0, 1500)))
            
            # Generate Document Content
            clauses = []
            num_clauses = random.randint(5, 12)
            for j in range(num_clauses):
                clause_text = [
                    f"Clause {j+1}.1: The parties hereby agree to the terms and conditions set forth in this section.",
                    f"Clause {j+1}.2: This agreement shall be governed by and construed in accordance with the laws of {jurisdiction}.",
                    f"Clause {j+1}.3: Any dispute arising out of or in connection with this agreement shall be referred to arbitration.",
                    f"Clause {j+1}.4: The duration of this agreement shall be {random.randint(1, 10)} years from the Effective Date.",
                    f"Clause {j+1}.5: Receiving Party shall hold the Confidential Information in strict confidence."
                ]
                clauses.append(random.choice(clause_text))
            
            content = "\\n\\n".join(clauses)
            
            documents.append({
                'document_id': f"DOC_{i:06d}",
                'title': title,
                'type': doc_type,
                'jurisdiction': jurisdiction,
                'date': date.strftime("%Y-%m-%d"),
                'content': content,
                'metadata': {
                    'word_count': len(content.split()),
                    'clause_count': num_clauses,
                    'created_at': datetime.now().isoformat()
                }
            })
        
        logger.info("Legal documents generated successfully")
        return documents
    
    def save_data(self, output_dir: str = "data/synthetic"):
        """
        Save all generated synthetic data to the output directory.
        
        Args:
            output_dir: Directory to save the data.
        """
        logger.info(f"Saving synthetic data to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Customer Data
        customer_df = self.generate_customer_data(1000)
        customer_df.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
        
        # Transaction Data
        transaction_df = self.generate_transaction_data(customer_df['customer_id'].tolist(), 5000)
        transaction_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
        
        # Product Data
        product_df = self.generate_product_data(500)
        product_df.to_csv(os.path.join(output_dir, "products.csv"), index=False)
        
        # Medical Data
        medical_df = self.generate_medical_records(1000)
        medical_df.to_csv(os.path.join(output_dir, "medical_records.csv"), index=False)
        
        # Legal Data
        legal_docs = self.generate_legal_documents(100)
        with open(os.path.join(output_dir, "legal_documents.json"), 'w', encoding='utf-8') as f:
            json.dump(legal_docs, f, ensure_ascii=False, indent=2)
        
        logger.info("All synthetic data saved successfully")

def main():
    """Main function execution."""
    parser = argparse.ArgumentParser(description='Generate synthetic data for AI Engineer Toolkit')
    parser.add_argument('--output_dir', type=str, default='data/synthetic', 
                        help='Output directory for synthetic data')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Base sample size for datasets')
    
    args = parser.parse_args()
    
    logger.info("Starting synthetic data generation...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    generator = SyntheticDataGenerator(seed=args.seed)
    generator.save_data(args.output_dir)
    
    logger.info("Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()
