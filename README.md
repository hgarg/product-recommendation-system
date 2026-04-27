# Product Recommendation System

Product recommendation system built with Python and Streamlit.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Datasets

This project uses two CSV files which are located in the repository root:

- `interactions.csv` — user-item interaction records
- `products.csv` — product metadata

You can find these files in this GitHub repo at the project root (paths: `./interactions.csv` and `./products.csv`).

## Using the app — upload & UI navigation

1. Start the app (check Help & Guidance at left panel of the streamlit application).
2. In the user interface use the file uploader to upload both `interactions.csv` and `products.csv`.
3. After uploading the files, navigate the UI using the buttons in this order:
	 - Load Data
	 - Explore
	 - Clean Data
	 - Run Model
	 - Results
4. The Results page will display model outputs and evaluation metrics.

Please reach out to me in case of any questions or issues with this application.