# Novartis Datathon
Ramon and Adrià participated in the Novartis 2024 dathathon.

During their preparation phase, they produced some code to solve the challenge for 2023, hoping that they can re-use it later.

## To do list --> Preparation phase.

### Work environment.
- conda environment. (create and update methodology).
- github branches. (producte)
- visual studio live share

### Preprocessing.
Recieves dataset, returns a pandas_df
- Assure column formats (date --> datetime...).
- Missing value imputation (lvl1 --> Nan solving).
- Data visualization (lvl1).

- Clustering (decomposition + elbow method)

### Forecasting.
Receives pandas_df, returns final parquet/csv.

## Useful comands & info:
We are running python 3.12.7 version.

If you install a new library in your pip environment, remember to update the requirements file by running:

```sh
pip freeze > requirements.txt
```

If you want to install the libraries from the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

Video with the [finalists' presentations](https://drive.google.com/file/d/1vLMugvAMAaC8un7TkRFNr4bUbGPRMexC/view?usp=sharing)