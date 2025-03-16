from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA

app = Flask(__name__)

# Create a static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/', methods=['GET', 'POST'])
def index():
    lasso_html = None
    ridge_html = None
    elasticnet_html = None
    pca_html = None
    pca_plot_path = None

    plot_paths = []

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']

        # If the user does not select a file, submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template('index.html', error=f'Error reading file: {str(e)}')

        # Assuming the first column is gene names and the last column is the target
        gene_names = df.iloc[:, 0]  # Store gene names
        X = df.iloc[:, 1:-1]  # Features: all columns except the first and last
        y = df.iloc[:, -1]  # Target: last column

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Determine which methods to use
        use_all = 'all' in request.form
        use_lasso = 'lasso' in request.form or use_all
        use_ridge = 'ridge' in request.form or use_all
        use_elasticnet = 'elasticnet' in request.form or use_all
        use_pca = 'pca' in request.form or use_all

        # Perform regression based on selected methods
        results = {}
        if use_lasso:
            lasso = Lasso(alpha=0.1)
            lasso.fit(X_scaled_df, y)
            lasso_df = pd.DataFrame({'Gene Name': X.columns, 'Lasso': lasso.coef_})
            lasso_html = lasso_df.to_html(classes='table table-striped table-bordered', index=False)
            results['Lasso'] = lasso.coef_

        if use_ridge:
            ridge = Ridge(alpha=0.1)
            ridge.fit(X_scaled_df, y)
            ridge_df = pd.DataFrame({'Gene Name': X.columns, 'Ridge': ridge.coef_})
            ridge_html = ridge_df.to_html(classes='table table-striped table-bordered', index=False)
            results['Ridge'] = ridge.coef_

        if use_elasticnet:
            elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
            elasticnet.fit(X_scaled_df, y)
            elasticnet_df = pd.DataFrame({'Gene Name': X.columns, 'ElasticNet': elasticnet.coef_})
            elasticnet_html = elasticnet_df.to_html(classes='table table-striped table-bordered', index=False)
            results['ElasticNet'] = elasticnet.coef_

        if use_pca:
            # Apply PCA - get 2 principal components
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)

            # Create DataFrame for PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

            # Add sample names if available (using row indices if not)
            if 'Sample' in df.columns:
                pca_df['Sample'] = df['Sample']
            else:
                pca_df['Sample'] = [f'Sample {i + 1}' for i in range(len(pca_df))]

            # Prepare the table HTML
            pca_table_df = pd.DataFrame({
                'Sample': pca_df['Sample'],
                'PC1': pca_df['PC1'],
                'PC2': pca_df['PC2']
            })
            pca_html = pca_table_df.to_html(classes='table table-striped table-bordered', index=False)

            # Create PC1 vs PC2 plot
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.8, edgecolors='w')

            # Add labels
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
            plt.title('PCA: PC1 vs PC2')

            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save plot as a static file
            plot_filename = 'pca_plot.png'
            plot_path = os.path.join('static', plot_filename)
            plt.savefig(plot_path)
            plt.close()  # Close the figure to free memory

            # Pass the plot path to the template
            pca_plot_path = '/' + plot_path

        # Store the results and corresponding DataFrames in the app object for download
        app.results = results
        app.X_columns = X.columns

        return render_template('index.html',
                               lasso_html=lasso_html,
                               ridge_html=ridge_html,
                               elasticnet_html=elasticnet_html,
                               pca_html=pca_html,
                               pca_plot_path=pca_plot_path,
                               form_submitted=True)

    return render_template('index.html', form_submitted=False)


@app.route('/download/<filetype>')
def download_results(filetype):
    # Retrieve data from the app object
    results = getattr(app, 'results', None)
    X_columns = getattr(app, 'X_columns', None)

    if results is None or X_columns is None:
        return "No results to download.", 400

    # Build combined coefficients DataFrame
    coefficients_df = pd.DataFrame(results, index=X_columns)

    try:
        if filetype == 'xlsx':
            # Create an Excel file in memory
            excel_file = io.BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                coefficients_df.to_excel(writer, sheet_name='Regression Results', index=True)
            excel_file.seek(0)

            return send_file(excel_file, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                             download_name='regression_results.xlsx', as_attachment=True)

        else:
            return "Invalid file type specified.", 400

    except Exception as e:
        return f"Error creating or sending file: {str(e)}", 500


@app.route('/download_pca_plot')
def download_pca_plot():
    """Download the PCA plot as a JPG file."""
    plot_path = os.path.join('static', 'pca_plot.png')
    try:
        return send_file(plot_path, mimetype='image/jpeg', as_attachment=True, download_name="pca_plot.png")
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    # Ensure a secret key is set (required for session usage)
    app.secret_key = os.urandom(24)  # Generate a random secret key
    app.run(debug=True)