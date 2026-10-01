# Movie Success Predictor

An interactive Streamlit app that predicts one of three IMDb score categories from movie metadata:

- **Flop:** IMDb score below 3.0
- **Average:** IMDb score from 3.0 up to 6.0
- **Hit:** IMDb score of 6.0 or higher

The project fulfils the assignment requirements with exploratory data analysis, data preprocessing, class creation, algorithm comparison, Random Forest evaluation, visualisations, and an interactive prediction page.

## What is in this project

```text
movie-success-predictor/
├── app.py                         # Streamlit dashboard and ML pipeline
├── Movie_Success_Analysis.ipynb    # Step-by-step Jupyter Notebook analysis
├── requirements.txt               # Packages Streamlit Cloud installs
├── data/movie_metadata.xlsx       # Supplied source dataset
├── PRESENTATION_GUIDE.md          # Slide-by-slide explanation
└── .gitignore
```

## Run it on your computer

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment:

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Open the local address printed in the terminal, usually `http://localhost:8501`.

## Run the Jupyter Notebook

1. Install Jupyter if you do not already have it: `pip install jupyter`.
2. From this project folder, run `jupyter notebook`.
3. Open `Movie_Success_Analysis.ipynb`.
4. Run the cells in order, from top to bottom. The notebook includes an optional package-installation cell, EDA, preprocessing, model comparison, Random Forest evaluation, and a sample prediction.

## Deploy on Streamlit Community Cloud

### 1. Put the project on GitHub

1. Sign in to [GitHub](https://github.com) and select **New repository**.
2. Name it `movie-success-predictor` and choose **Public** for the easiest Community Cloud deployment.
3. Do not create a README on GitHub because this folder already includes one.
4. Upload every item in this project folder, including the `data` folder and `requirements.txt`, then commit the files. You can also use Git commands:

   ```bash
   git init
   git add .
   git commit -m "Add Movie Success Predictor"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/movie-success-predictor.git
   git push -u origin main
   ```

### 2. Create the Streamlit app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Select **Create app**.
3. Choose your repository, the `main` branch, and set the main file path to `app.py`.
4. Select **Deploy**.
5. Wait for the package installation and first run to finish. The service then gives you a public URL.

### 3. Check the deployed app

1. Open the public URL.
2. Confirm the **Overview** page shows 4,998 movies after duplicate removal.
3. Open **Model performance** and wait for the model to train once.
4. Test the **Predict a movie** page and make sure a class and three probabilities appear.
5. Copy the public URL into your report or presentation.

### If deployment fails

- If Streamlit says it cannot find the spreadsheet, confirm this exact repository path exists: `data/movie_metadata.xlsx`.
- If a package cannot be installed, check that `requirements.txt` is in the top-level project folder beside `app.py`.
- Open **Manage app** → **Logs** in Streamlit Community Cloud. The first meaningful error line normally identifies the missing package or incorrect path.
- After changing a GitHub file, commit and push it. Streamlit Community Cloud normally redeploys from the updated repository automatically.

## Design choices you should explain

The original data contains 5,043 rows. The app removes 45 exact duplicates, leaving 4,998 rows. It uses a stratified 80/20 train-test split so each success class appears in both sets. It uses macro F1 alongside accuracy because Flop is a rare class.

The main model intentionally excludes gross revenue, review counts, vote counts, and movie-page likes. These variables are only known after release, so including them would create data leakage and produce overly optimistic results. The model instead uses attributes that can be known before release, such as planned duration, budget, cast popularity, country, language, rating, and genre.

Read [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) before presenting.
