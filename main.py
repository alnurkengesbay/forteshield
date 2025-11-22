import argparse
import logging
import sys
from src import config
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.model import FraudModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / "app.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection System")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Mode: train or predict')
    args = parser.parse_args()

    try:
        # 1. Load Data
        loader = DataLoader()
        df = loader.load_and_merge()

        # 2. Feature Engineering
        engineer = FeatureEngineer()
        df_processed = engineer.preprocess(df)

        # 3. Model Operations
        model = FraudModel()

        if args.mode == 'train':
            logger.info("Starting Training Pipeline...")
            model.train(df_processed)
            model.save()
            
            # Demonstrate Explainability on a sample fraud case from training data (just for demo)
            # In real life, this would be on new data
            fraud_indices = df_processed[df_processed[config.TARGET_COL] == 1].index
            if not fraud_indices.empty:
                # We need to pass the processed dataframe but reset index to match iloc logic in explain_prediction
                # Or just pass the row. The explain_prediction method expects a DF and an integer index.
                # Let's just pick the first fraud case found in the processed data.
                # Note: model.train splits data, so we are explaining on the full set here which is fine for demo.
                idx_to_explain = df_processed.index.get_loc(fraud_indices[0])
                model.explain_prediction(df_processed, index=idx_to_explain)

        elif args.mode == 'predict':
            logger.info("Starting Prediction Pipeline...")
            model.load()
            probs = model.predict(df_processed)
            df_processed['fraud_probability'] = probs
            
            # Save results
            output_path = config.DATA_DIR / "predictions.csv"
            df_processed[[config.ID_COL, 'fraud_probability']].to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
